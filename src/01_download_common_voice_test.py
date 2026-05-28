#!/usr/bin/env python3
"""Common Voice 日本語 test split をダウンロードし metadata.csv を作成する.

Phase1 step 01.

実行例:
    uv run python src/01_download_common_voice_test.py --seed 42

仕様:
- HuggingFace からデータセット (デフォルト: fixie-ai/common_voice_17_0) の ja split を読む
- データセット全体を save_to_disk で保存 (個別 clip ファイルは作らない)
- `path` でソートしたあと `--seed` で shuffle し安定した sample_id 順序を作る
- 全件を metadata.csv / test.tsv に書く
- metadata.csv に既に同じ sample_id があればその行は再書き込みしない
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

DEFAULT_DATASET_NAME = "fixie-ai/common_voice_17_0"
DEFAULT_DATASET_VERSION = "17_0"
DEFAULT_LOCALE = "ja"
DEFAULT_SPLIT = "test"
DEFAULT_OUTPUT_DIR = Path("inputs/common_voice_ja_test")

METADATA_FIELDS = [
    "sample_id",
    "dataset",
    "dataset_version",
    "split",
    "reference_text",
    "original_audio_path",
    "duration_sec",
]

TSV_FIELDS = [
    "sample_id",
    "client_id",
    "path",
    "sentence",
    "duration_sec",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=42, help="シャッフル seed")
    p.add_argument(
        "--dataset-name",
        default=DEFAULT_DATASET_NAME,
        help="HF データセット名. 例: fixie-ai/common_voice_17_0",
    )
    p.add_argument(
        "--dataset-version",
        default=DEFAULT_DATASET_VERSION,
        help="metadata に記録するバージョン文字列. 例: 17_0",
    )
    p.add_argument("--locale", default=DEFAULT_LOCALE)
    p.add_argument("--split", default=DEFAULT_SPLIT)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument(
        "--hf-token",
        default=None,
        help="Hugging Face token. 省略時は HF_TOKEN 環境変数を使用",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="データセット保存と metadata.csv を上書き",
    )
    return p.parse_args()


def load_existing_sample_ids(metadata_path: Path) -> set[str]:
    if not metadata_path.exists():
        return set()
    with metadata_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {row["sample_id"] for row in reader if row.get("sample_id")}


def main() -> int:
    args = parse_args()

    try:
        from datasets import Audio, load_dataset, load_from_disk
    except ImportError:
        print(
            "datasets パッケージがありません。"
            "`uv add datasets soundfile librosa tqdm` を実行してください。",
            file=sys.stderr,
        )
        return 1

    try:
        from tqdm import tqdm
    except ImportError:

        def tqdm(it, **_kw):  # type: ignore[no-redef]
            return it

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    dataset_name = args.dataset_name

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = output_dir / "dataset"

    # --- データセットのダウンロードと保存 ---
    if dataset_dir.exists() and not args.overwrite:
        print(f"[skip] saved dataset found: {dataset_dir}")
        ds = load_from_disk(str(dataset_dir))
    else:
        print(f"[load] {dataset_name} locale={args.locale} split={args.split}")
        ds = load_dataset(
            dataset_name,
            args.locale,
            split=args.split,
            token=hf_token,
        )

        # 順序を再現可能にするため path でソート→ shuffle
        if "path" in ds.column_names:
            print("[order] sort by path")
            ds = ds.sort("path")
        print(f"[order] shuffle seed={args.seed}")
        ds = ds.shuffle(seed=args.seed)

        # 音声は生バイトのまま保存 (コンパクト)
        ds_raw = ds.cast_column("audio", Audio(decode=False))
        ds_raw.save_to_disk(str(dataset_dir))
        print(f"[save] dataset ({len(ds)} samples) -> {dataset_dir}")

    print(f"[info] total samples: {len(ds)}")

    # --- metadata.csv / test.tsv 作成 ---
    metadata_path = output_dir / "metadata.csv"
    tsv_path = output_dir / "test.tsv"

    if args.overwrite and metadata_path.exists():
        metadata_path.unlink()

    existing_sample_ids = load_existing_sample_ids(metadata_path)
    if existing_sample_ids:
        print(f"[skip] existing metadata rows: {len(existing_sample_ids)}")

    file_exists = metadata_path.exists()
    mode = "a" if file_exists else "w"

    # 音声のデコードが必要 (duration 計算用)
    if ds.features.get("audio") and hasattr(ds.features["audio"], "decode"):
        ds_decoded = ds.cast_column("audio", Audio(decode=True))
    else:
        ds_decoded = ds

    tsv_rows: list[dict[str, str]] = []
    n_new = 0
    n_skipped = 0

    with metadata_path.open(mode, newline="", encoding="utf-8") as fmeta:
        writer = csv.DictWriter(fmeta, fieldnames=METADATA_FIELDS)
        if not file_exists:
            writer.writeheader()

        for idx in tqdm(range(len(ds)), desc="metadata"):
            sample_id = f"cv_ja_{args.split}_{idx + 1:06d}"
            row = ds_decoded[idx]

            audio = row.get("audio") or {}
            audio_path = row.get("path") or audio.get("path") or ""
            sentence = row.get("sentence") or ""

            audio_array = audio.get("array")
            sr = audio.get("sampling_rate")
            if audio_array is not None and sr:
                duration_sec = float(len(audio_array)) / float(sr)
            else:
                duration_sec = None

            tsv_rows.append(
                {
                    "sample_id": sample_id,
                    "client_id": row.get("client_id") or "",
                    "path": audio_path,
                    "sentence": sentence,
                    "duration_sec": (
                        "" if duration_sec is None else f"{duration_sec:.3f}"
                    ),
                },
            )

            if sample_id in existing_sample_ids:
                n_skipped += 1
                continue

            writer.writerow(
                {
                    "sample_id": sample_id,
                    "dataset": dataset_name,
                    "dataset_version": args.dataset_version,
                    "split": args.split,
                    "reference_text": sentence,
                    "original_audio_path": audio_path,
                    "duration_sec": (
                        "" if duration_sec is None else f"{duration_sec:.6f}"
                    ),
                },
            )
            n_new += 1

    with tsv_path.open("w", newline="", encoding="utf-8") as ftsv:
        tsv_writer = csv.DictWriter(ftsv, fieldnames=TSV_FIELDS, delimiter="\t")
        tsv_writer.writeheader()
        for r in tsv_rows:
            tsv_writer.writerow(r)

    print(f"[done] new={n_new} skipped={n_skipped} total={len(ds)}")
    print(f"[done] metadata.csv: {metadata_path}")
    print(f"[done] test.tsv:    {tsv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
