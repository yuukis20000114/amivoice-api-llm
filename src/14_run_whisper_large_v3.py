#!/usr/bin/env python3
"""Whisper large-v3 で音声認識を実行し CSV を出力する.

Phase1 ASR スクリプト.

実行例:
    run_gpu uv run python src/14_run_whisper_large_v3.py --variant clean
    run_gpu uv run python src/14_run_whisper_large_v3.py --variant white_snr_10
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import soundfile as sf
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))
from asr_csv import load_existing_keys, open_asr_csv

MODEL_ID = "openai/whisper-large-v3"
DEFAULT_METADATA = Path("inputs/common_voice_ja_test/metadata.csv")
DEFAULT_AUDIO_DIR = Path("inputs/audio_variants")
DEFAULT_OUTPUT_DIR = Path("outputs")
DEFAULT_RAW_BASE = Path("outputs/raw_responses/whisper_large_v3")
DEFAULT_BATCH_SIZE = 4


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
        args.output = DEFAULT_OUTPUT_DIR / f"asr_whisper_large_v3_{args.variant}.csv"
    if args.raw_dir is None:
        args.raw_dir = DEFAULT_RAW_BASE / args.variant
    return args


def load_model() -> pipeline:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(MODEL_ID)

    return pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=dtype,
        device=device,
    )


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
        "asr_provider": "openai_whisper",
        "asr_model": MODEL_ID,
        "asr_text": "",
        "status": "",
        "error_message": "",
        "processing_time_sec": "",
        "raw_response_path": "",
    }


def main() -> int:  # noqa: C901, PLR0912, PLR0915
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

    valid_pending: list[dict[str, str]] = []
    error_rows: list[dict[str, str]] = []
    for meta_row in pending:
        audio_path = variant_dir / f"{meta_row['sample_id']}.wav"
        if audio_path.exists():
            valid_pending.append(meta_row)
        else:
            row = _make_row(meta_row, variant, audio_path)
            row["status"] = "error"
            row["error_message"] = "audio file not found"
            error_rows.append(row)

    print(f"[model] loading {MODEL_ID} ...")
    pipe = load_model()
    print(f"[model] loaded on {pipe.device}")

    generate_kwargs = {"language": "ja", "task": "transcribe"}
    print(
        f"[run] variant={variant} model={MODEL_ID} "
        f"device={pipe.device} batch_size={batch_size}",
    )

    csv_file, csv_writer = open_asr_csv(args.output)
    n_ok = 0
    n_error = 0

    try:
        for row in error_rows:
            csv_writer.writerow(row)
            n_error += 1
        if error_rows:
            csv_file.flush()

        pbar = tqdm(
            total=len(pending),
            initial=len(error_rows),
            desc=f"whisper_v3/{variant}",
        )
        with torch.inference_mode():
            for chunk_start in range(0, len(valid_pending), batch_size):
                chunk = valid_pending[chunk_start : chunk_start + batch_size]

                audio_inputs = []
                for meta_row in chunk:
                    audio_path = variant_dir / f"{meta_row['sample_id']}.wav"
                    array, sr = sf.read(str(audio_path), dtype="float32")
                    audio_inputs.append({"raw": array, "sampling_rate": sr})

                t0 = time.monotonic()
                results = pipe(
                    audio_inputs,
                    batch_size=batch_size,
                    generate_kwargs=generate_kwargs,
                )
                elapsed = time.monotonic() - t0
                per_sample = elapsed / len(results)

                for meta_row, result in zip(chunk, results, strict=True):
                    sample_id = meta_row["sample_id"]
                    audio_path = variant_dir / f"{sample_id}.wav"
                    raw_path = raw_dir / f"{sample_id}.json"

                    raw_path.write_text(
                        json.dumps(result, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )

                    row = _make_row(meta_row, variant, audio_path)
                    row["status"] = "ok"
                    row["asr_text"] = result.get("text", "")
                    row["processing_time_sec"] = f"{per_sample:.3f}"
                    row["raw_response_path"] = str(raw_path)

                    csv_writer.writerow(row)
                    n_ok += 1

                csv_file.flush()
                pbar.update(len(chunk))

        pbar.close()
    finally:
        csv_file.close()
        del pipe
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
