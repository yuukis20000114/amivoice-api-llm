#!/usr/bin/env python3
"""Common Voice 音声に白色ノイズを付与し audio_variants/ を作成する.

Phase1 step 02.

実行例:
    uv run python src/02_add_white_noise.py --snr-db 30 20 10 1 --seed 42

仕様:
- 入力: ステップ01 で保存したデータセット + metadata.csv
- 出力: `inputs/audio_variants/{variant}/{sample_id}.wav`
  - `clean`: 元音声を 16kHz mono WAV に変換しただけ
  - `white_snr_XX`: 指定 SNR (dB) で白色ノイズを付与
- ノイズは (base_seed, sample_id, snr_db) から派生した seed で決定的に生成
- 既存 WAV があれば `--overwrite` 指定が無い限りスキップ
- 混合後の peak が 1.0 を超える場合のみピーク正規化 (SNR は不変)
- 無音に近い音声 (signal_power < SIGNAL_POWER_FLOOR) はスキップして警告
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

DEFAULT_DATASET_DIR = Path("inputs/common_voice_ja_test/dataset")
DEFAULT_METADATA = Path("inputs/common_voice_ja_test/metadata.csv")
DEFAULT_OUTPUT_DIR = Path("inputs/audio_variants")
DEFAULT_SNR_DB = [30.0, 20.0, 10.0, 1.0]
DEFAULT_SAMPLE_RATE = 16000
SIGNAL_POWER_FLOOR = 1e-10
PEAK_NORM_TARGET = 0.99


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="ステップ01 で save_to_disk したデータセットのパス",
    )
    p.add_argument(
        "--metadata",
        type=Path,
        default=DEFAULT_METADATA,
        help="ステップ01 で作成した metadata.csv のパス",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="audio_variants の出力ディレクトリ",
    )
    p.add_argument(
        "--snr-db",
        nargs="+",
        type=float,
        default=DEFAULT_SNR_DB,
        help="付与する SNR (dB) のリスト. 例: --snr-db 30 20 10 1",
    )
    p.add_argument(
        "--sample-rate",
        type=int,
        default=DEFAULT_SAMPLE_RATE,
        help="出力音声のサンプリングレート (Hz)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="ノイズ生成 seed のベース値",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="既存の WAV を上書きする",
    )
    return p.parse_args()


def variant_dir_name(snr_db: float) -> str:
    if abs(snr_db - round(snr_db)) < 1e-9:
        return f"white_snr_{int(round(snr_db))}"
    sign = "neg" if snr_db < 0 else ""
    return "white_snr_" + sign + str(abs(snr_db)).replace(".", "p")


def derive_seed(base_seed: int, sample_id: str, snr_db: float) -> int:
    key = f"{base_seed}|{sample_id}|{snr_db:.6f}".encode()
    digest = hashlib.sha256(key).digest()
    return int.from_bytes(digest[:4], "big")


def resample_mono(audio_array: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if audio_array.ndim > 1:
        audio_array = np.mean(audio_array, axis=-1)
    if orig_sr != target_sr:
        audio_array = librosa.resample(
            audio_array.astype(np.float32), orig_sr=orig_sr, target_sr=target_sr,
        )
    return audio_array.astype(np.float32)


def write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    sf.write(str(path), audio, sample_rate, subtype="PCM_16")


def mix_white_noise(
    signal: np.ndarray,
    snr_db: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, bool]:
    signal_power = float(np.mean(np.square(signal, dtype=np.float64)))
    if signal_power < SIGNAL_POWER_FLOOR:
        msg = f"signal_power={signal_power:.3e} が小さすぎます (無音判定)"
        raise ValueError(msg)

    noise_power_target = signal_power / (10.0 ** (snr_db / 10.0))
    noise = rng.standard_normal(signal.shape)
    noise_rms = float(np.sqrt(np.mean(np.square(noise))))
    scale = math.sqrt(noise_power_target) / noise_rms
    noise = (noise * scale).astype(np.float32)

    mixed = signal.astype(np.float32) + noise
    peak = float(np.max(np.abs(mixed)))
    normalized = False
    if peak > 1.0:
        mixed = (mixed / peak) * PEAK_NORM_TARGET
        normalized = True
    return mixed.astype(np.float32), normalized


def main() -> int:
    args = parse_args()

    try:
        from datasets import Audio, load_from_disk
    except ImportError:
        print(
            "datasets パッケージがありません。"
            "`uv add datasets` を実行してください。",
            file=sys.stderr,
        )
        return 1

    try:
        from tqdm import tqdm
    except ImportError:

        def tqdm(it, **_kw):  # type: ignore[no-redef]
            return it

    metadata_path: Path = args.metadata
    if not metadata_path.exists():
        print(f"metadata.csv が見つかりません: {metadata_path}", file=sys.stderr)
        return 1

    dataset_dir: Path = args.dataset_dir
    if not dataset_dir.exists():
        print(
            f"保存済みデータセットが見つかりません: {dataset_dir}\n"
            "先に src/01_download_common_voice_test.py を実行してください。",
            file=sys.stderr,
        )
        return 1

    # データセットを音声デコード付きでロード
    print(f"[load] dataset: {dataset_dir}")
    ds = load_from_disk(str(dataset_dir))
    ds = ds.cast_column("audio", Audio(decode=True))
    print(f"[info] dataset rows: {len(ds)}")

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    clean_dir = output_dir / "clean"
    clean_dir.mkdir(parents=True, exist_ok=True)

    variant_dirs: dict[float, Path] = {}
    for snr in args.snr_db:
        d = output_dir / variant_dir_name(snr)
        d.mkdir(parents=True, exist_ok=True)
        variant_dirs[snr] = d

    with metadata_path.open(newline="", encoding="utf-8") as f:
        meta_rows = list(csv.DictReader(f))
    print(f"[load] metadata rows: {len(meta_rows)}")
    print(f"[variants] clean + {[variant_dir_name(s) for s in args.snr_db]}")

    n_skipped_clean = 0
    n_written_clean = 0
    n_skipped_noisy = 0
    n_written_noisy = 0
    n_failed = 0
    n_peaknorm = 0
    n_skipped_all = 0

    for meta_row in tqdm(meta_rows, desc="mix"):
        sample_id = meta_row["sample_id"]

        # sample_id "cv_ja_test_000001" -> dataset index 0
        try:
            ds_idx = int(sample_id.rsplit("_", 1)[1]) - 1
        except (ValueError, IndexError):
            print(f"[warn] invalid sample_id: {sample_id}", file=sys.stderr)
            n_failed += 1
            continue

        if ds_idx < 0 or ds_idx >= len(ds):
            print(f"[warn] index out of range: {sample_id} -> {ds_idx}", file=sys.stderr)
            n_failed += 1
            continue

        clean_out = clean_dir / f"{sample_id}.wav"
        noisy_outs: dict[float, Path] = {
            snr: vdir / f"{sample_id}.wav" for snr, vdir in variant_dirs.items()
        }
        all_targets = [clean_out, *noisy_outs.values()]
        if not args.overwrite and all(p.exists() for p in all_targets):
            n_skipped_all += 1
            continue

        try:
            audio = ds[ds_idx]["audio"]
            audio_array = audio["array"]
            orig_sr = audio["sampling_rate"]
            signal = resample_mono(audio_array, orig_sr, args.sample_rate)
        except Exception as e:  # noqa: BLE001
            print(f"[warn] decode failed {sample_id}: {e}", file=sys.stderr)
            n_failed += 1
            continue

        if args.overwrite or not clean_out.exists():
            write_wav(clean_out, signal, args.sample_rate)
            n_written_clean += 1
        else:
            n_skipped_clean += 1

        for snr_db, noisy_out in noisy_outs.items():
            if not args.overwrite and noisy_out.exists():
                n_skipped_noisy += 1
                continue
            seed = derive_seed(args.seed, sample_id, snr_db)
            rng = np.random.default_rng(seed)
            try:
                mixed, normalized = mix_white_noise(signal, snr_db, rng)
            except ValueError as e:
                print(
                    f"[warn] {sample_id} snr={snr_db}: {e}",
                    file=sys.stderr,
                )
                n_failed += 1
                continue
            if normalized:
                n_peaknorm += 1
            write_wav(noisy_out, mixed, args.sample_rate)
            n_written_noisy += 1

    print(
        f"[done] clean: written={n_written_clean} skipped={n_skipped_clean} "
        f"| noisy: written={n_written_noisy} skipped={n_skipped_noisy} "
        f"peaknorm={n_peaknorm} | skipped_all={n_skipped_all} failed={n_failed}",
    )
    print(f"[done] output dir: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
