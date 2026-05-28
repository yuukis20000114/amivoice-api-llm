#!/usr/bin/env python3
"""ReazonSpeech NeMo v2 で音声認識を実行し CSV を出力する.

Phase1 ASR スクリプト.

実行例:
    run_gpu uv run python src/17_run_reazonspeech.py --variant clean
    run_gpu uv run python src/17_run_reazonspeech.py --variant white_snr_10
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from asr_csv import load_existing_keys, open_asr_csv

MODEL_ID = "reazonspeech-nemo-v2"
DEFAULT_METADATA = Path("inputs/common_voice_ja_test/metadata.csv")
DEFAULT_AUDIO_DIR = Path("inputs/audio_variants")
DEFAULT_OUTPUT_DIR = Path("outputs")
DEFAULT_RAW_BASE = Path("outputs/raw_responses/reazonspeech")
DEFAULT_BATCH_SIZE = 8

SAMPLERATE = 16000
PAD_SECONDS = 0.5


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--variant",
        required=True,
        help="処理する variant 名 (例: clean, white_snr_10)",
    )
    p.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    p.add_argument("--audio-dir", type=Path, default=DEFAULT_AUDIO_DIR)
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="出力 CSV パス",
    )
    p.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help="raw response 保存先",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="バッチサイズ (デフォルト: %(default)s)",
    )
    args = p.parse_args()
    if args.output is None:
        args.output = DEFAULT_OUTPUT_DIR / f"asr_reazonspeech_{args.variant}.csv"
    if args.raw_dir is None:
        args.raw_dir = DEFAULT_RAW_BASE / args.variant
    return args


def load_model():
    from reazonspeech.nemo.asr import load_model as _load  # noqa: PLC0415

    return _load(device="cuda" if torch.cuda.is_available() else "cpu")


def _load_and_preprocess(audio_path: Path) -> torch.Tensor:
    """WAV を読み込み、正規化・パディングしてテンソルを返す."""
    array, sr = sf.read(str(audio_path), dtype="float32")
    if sr != SAMPLERATE:
        import librosa  # noqa: PLC0415

        array = librosa.resample(array, orig_sr=sr, target_sr=SAMPLERATE)
    if len(array.shape) > 1:
        array = array.mean(axis=1)
    pad_width = int(PAD_SECONDS * SAMPLERATE)
    array = np.pad(array, pad_width=pad_width, mode="constant")
    return torch.from_numpy(array)


def _make_row(
    meta_row: dict[str, str],
    variant: str,
    audio_path: Path,
) -> dict[str, str]:
    return {
        "sample_id": meta_row["sample_id"],
        "audio_variant": variant,
        "audio_path": str(audio_path),
        "reference_text": meta_row.get("reference_text", ""),
        "asr_provider": "reazon",
        "asr_model": MODEL_ID,
        "asr_text": "",
        "status": "",
        "error_message": "",
        "processing_time_sec": "",
        "raw_response_path": "",
    }


def _decode_and_save(  # noqa: PLR0913
    model: object,
    hyp: object,
    meta_row: dict[str, str],
    variant: str,
    audio_path: Path,
    raw_dir: Path,
    elapsed_per_sample: float,
) -> dict[str, str]:
    from reazonspeech.nemo.asr.decode import decode_hypothesis  # noqa: PLC0415

    row = _make_row(meta_row, variant, audio_path)
    result = decode_hypothesis(model, hyp)

    raw_data = {
        "text": result.text,
        "segments": [
            {
                "text": seg.text,
                "start": seg.start_seconds,
                "end": seg.end_seconds,
            }
            for seg in (result.segments or [])
        ],
    }

    raw_path = raw_dir / f"{meta_row['sample_id']}.json"
    raw_path.write_text(
        json.dumps(raw_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    row["status"] = "ok"
    row["asr_text"] = result.text
    row["processing_time_sec"] = f"{elapsed_per_sample:.3f}"
    row["raw_response_path"] = str(raw_path)
    return row


def _process_sample(
    meta_row: dict[str, str],
    variant: str,
    variant_dir: Path,
    raw_dir: Path,
    model: object,
) -> dict[str, str]:
    """単一サンプル処理 (フォールバック用)."""
    sample_id = meta_row["sample_id"]
    audio_path = variant_dir / f"{sample_id}.wav"
    row = _make_row(meta_row, variant, audio_path)

    if not audio_path.exists():
        row["status"] = "error"
        row["error_message"] = "audio file not found"
        return row

    t0 = time.monotonic()
    try:
        waveform = _load_and_preprocess(audio_path)
        results = model.transcribe(
            [waveform],
            batch_size=1,
            return_hypotheses=True,
            verbose=False,
        )
        hyp = results[0]
        elapsed = time.monotonic() - t0
        return _decode_and_save(
            model,
            hyp,
            meta_row,
            variant,
            audio_path,
            raw_dir,
            elapsed,
        )
    except (RuntimeError, ValueError, OSError) as e:
        row["status"] = "error"
        row["error_message"] = str(e)
        row["processing_time_sec"] = f"{time.monotonic() - t0:.3f}"
        return row


def _process_batch(  # noqa: PLR0913
    batch_meta: list[dict[str, str]],
    variant: str,
    variant_dir: Path,
    raw_dir: Path,
    model: object,
    batch_size: int,
) -> list[dict[str, str]]:
    waveforms: list[torch.Tensor] = []
    valid_indices: list[int] = []
    rows: list[dict[str, str]] = [None] * len(batch_meta)  # type: ignore[list-item]

    for i, meta_row in enumerate(batch_meta):
        audio_path = variant_dir / f"{meta_row['sample_id']}.wav"
        row = _make_row(meta_row, variant, audio_path)

        if not audio_path.exists():
            row["status"] = "error"
            row["error_message"] = "audio file not found"
            rows[i] = row
            continue

        try:
            waveform = _load_and_preprocess(audio_path)
        except (RuntimeError, ValueError, OSError) as e:
            row["status"] = "error"
            row["error_message"] = f"audio read failed: {e}"
            rows[i] = row
            continue

        waveforms.append(waveform)
        valid_indices.append(i)
        rows[i] = row

    if not waveforms:
        return rows

    t0 = time.monotonic()
    try:
        hypotheses = model.transcribe(
            waveforms,
            batch_size=batch_size,
            return_hypotheses=True,
            verbose=False,
        )
    except (RuntimeError, ValueError, OSError):
        for idx in valid_indices:
            rows[idx] = _process_sample(
                batch_meta[idx],
                variant,
                variant_dir,
                raw_dir,
                model,
            )
        return rows

    elapsed = time.monotonic() - t0
    per_sample = elapsed / len(hypotheses)

    for idx, hyp in zip(valid_indices, hypotheses, strict=True):
        meta_row = batch_meta[idx]
        audio_path = variant_dir / f"{meta_row['sample_id']}.wav"
        try:
            rows[idx] = _decode_and_save(
                model,
                hyp,
                meta_row,
                variant,
                audio_path,
                raw_dir,
                per_sample,
            )
        except (RuntimeError, ValueError, OSError) as e:
            rows[idx]["status"] = "error"
            rows[idx]["error_message"] = f"decode failed: {e}"
            rows[idx]["processing_time_sec"] = f"{per_sample:.3f}"

    return rows


def main() -> int:  # noqa: PLR0915
    args = parse_args()

    try:
        from tqdm import tqdm  # noqa: PLC0415
    except ImportError:

        def tqdm(it, **_kw):  # type: ignore[no-redef]
            return it

    variant: str = args.variant
    batch_size: int = args.batch_size
    variant_dir = args.audio_dir / variant
    if not variant_dir.exists():
        print(f"variant ディレクトリが見つかりません: {variant_dir}", file=sys.stderr)
        return 1

    metadata_path: Path = args.metadata
    if not metadata_path.exists():
        print(f"metadata.csv が見つかりません: {metadata_path}", file=sys.stderr)
        return 1

    with metadata_path.open(newline="", encoding="utf-8") as f:
        meta_rows = list(csv.DictReader(f))
    print(f"[load] metadata: {len(meta_rows)} samples")

    existing_keys = load_existing_keys(args.output)
    print(f"[skip] existing results: {len(existing_keys)}")

    pending = [
        row for row in meta_rows if (row["sample_id"], variant) not in existing_keys
    ]
    print(f"[pending] {len(pending)} samples to process")

    if not pending:
        print("[done] nothing to do")
        return 0

    raw_dir: Path = args.raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)

    print(f"[model] loading {MODEL_ID} ...")
    model = load_model()
    print("[model] loaded")

    print(f"[run] variant={variant} model={MODEL_ID} batch_size={batch_size}")

    csv_file, csv_writer = open_asr_csv(args.output)
    n_ok = 0
    n_error = 0

    try:
        pbar = tqdm(total=len(pending), desc=f"reazon/{variant}")
        with torch.inference_mode():
            for batch_start in range(0, len(pending), batch_size):
                batch_meta = pending[batch_start : batch_start + batch_size]

                batch_rows = _process_batch(
                    batch_meta,
                    variant,
                    variant_dir,
                    raw_dir,
                    model,
                    batch_size,
                )

                for row in batch_rows:
                    csv_writer.writerow(row)
                    if row["status"] == "ok":
                        n_ok += 1
                    else:
                        n_error += 1
                csv_file.flush()
                pbar.update(len(batch_meta))
        pbar.close()
    finally:
        csv_file.close()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(
        f"[done] ok={n_ok} error={n_error} skipped={len(existing_keys)} "
        f"total={len(meta_rows)}",
    )
    print(f"[done] output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
