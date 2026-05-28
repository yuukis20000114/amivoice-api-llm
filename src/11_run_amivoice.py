#!/usr/bin/env python3
"""AmiVoice Cloud Platform で音声認識を実行し CSV を出力する.

Phase1 ASR スクリプト.

実行例:
    uv run python src/11_run_amivoice.py --variant clean
    uv run python src/11_run_amivoice.py --variant white_snr_10
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import dotenv_values
from tenacity import (
    retry,
    retry_if_result,
    stop_after_attempt,
    wait_exponential,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from asr_csv import ASR_CSV_FIELDS, load_existing_keys, open_asr_csv

AMIVOICE_ENDPOINT = "https://acp-api.amivoice.com/v1/recognize"
DEFAULT_ENGINE = "-a2-ja-general"
DEFAULT_METADATA = Path("inputs/common_voice_ja_test/metadata.csv")
DEFAULT_AUDIO_DIR = Path("inputs/audio_variants")
DEFAULT_OUTPUT_DIR = Path("outputs")
DEFAULT_RAW_BASE = Path("outputs/raw_responses/amivoice")

RETRYABLE_CODES = {"b", "$"}


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
        help="出力 CSV パス (デフォルト: outputs/asr_amivoice_{variant}.csv)",
    )
    p.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help="raw response 保存先 (デフォルト: outputs/raw_responses/amivoice/{variant}/)",
    )
    p.add_argument(
        "--engine",
        default=None,
        help="AmiVoice エンジン (デフォルト: env AMIVOICE_ENGINE or -a2-ja-general)",
    )
    args = p.parse_args()
    if args.output is None:
        args.output = DEFAULT_OUTPUT_DIR / f"asr_amivoice_{args.variant}.csv"
    if args.raw_dir is None:
        args.raw_dir = DEFAULT_RAW_BASE / args.variant
    return args


def get_env(key: str, default: str | None = None) -> str | None:
    """os.environ → .env ファイルの順で探す."""
    val = os.environ.get(key)
    if val:
        return val
    env_file = Path(".env")
    if env_file.exists():
        vals = dotenv_values(env_file)
        val = vals.get(key)
    return val or default


def _is_retryable(result: dict) -> bool:
    return result.get("code", "") in RETRYABLE_CODES


@retry(
    retry=retry_if_result(_is_retryable),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    reraise=True,
)
def call_amivoice(
    audio_path: Path,
    api_key: str,
    engine: str,
) -> dict:
    with audio_path.open("rb") as af:
        resp = requests.post(
            AMIVOICE_ENDPOINT,
            data={
                "u": api_key,
                "d": f"grammarFileNames={engine}",
            },
            files={"a": af},
            timeout=120,
        )
    resp.raise_for_status()
    return resp.json()


def main() -> int:
    args = parse_args()

    api_key = get_env("AMIVOICE_API_KEY")
    if not api_key:
        print(
            "AMIVOICE_API_KEY が設定されていません。"
            ".env または環境変数で設定してください。",
            file=sys.stderr,
        )
        return 1

    engine = args.engine or get_env("AMIVOICE_ENGINE", DEFAULT_ENGINE)

    try:
        from tqdm import tqdm
    except ImportError:

        def tqdm(it, **_kw):  # type: ignore[no-redef]
            return it

    variant: str = args.variant
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

    raw_dir: Path = args.raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)

    print(f"[run] variant={variant} engine={engine} endpoint={AMIVOICE_ENDPOINT}")

    csv_file, csv_writer = open_asr_csv(args.output)
    n_ok = 0
    n_error = 0
    n_skipped = 0

    try:
        for meta_row in tqdm(meta_rows, desc=f"amivoice/{variant}"):
            sample_id = meta_row["sample_id"]
            reference_text = meta_row.get("reference_text", "")

            if (sample_id, variant) in existing_keys:
                n_skipped += 1
                continue

            audio_path = variant_dir / f"{sample_id}.wav"
            raw_path = raw_dir / f"{sample_id}.json"

            row: dict[str, str] = {
                "sample_id": sample_id,
                "audio_variant": variant,
                "audio_path": str(audio_path),
                "reference_text": reference_text,
                "asr_provider": "amivoice",
                "asr_model": engine,
                "asr_text": "",
                "status": "",
                "error_message": "",
                "processing_time_sec": "",
                "raw_response_path": str(raw_path),
            }

            if not audio_path.exists():
                row["status"] = "error"
                row["error_message"] = "audio file not found"
                row["raw_response_path"] = ""
                csv_writer.writerow(row)
                csv_file.flush()
                n_error += 1
                continue

            t0 = time.monotonic()
            try:
                result = call_amivoice(audio_path, api_key, engine)
            except requests.exceptions.RequestException as e:
                row["status"] = "error"
                row["error_message"] = str(e)
                row["processing_time_sec"] = f"{time.monotonic() - t0:.3f}"
                row["raw_response_path"] = ""
                csv_writer.writerow(row)
                csv_file.flush()
                n_error += 1
                continue
            elapsed = time.monotonic() - t0

            raw_path.write_text(
                json.dumps(result, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            code = result.get("code", "")
            if code:
                row["status"] = "error"
                row["error_message"] = f"{code}: {result.get('message', '')}"
                row["processing_time_sec"] = f"{elapsed:.3f}"
                csv_writer.writerow(row)
                csv_file.flush()
                n_error += 1
            else:
                row["status"] = "ok"
                row["asr_text"] = result.get("text", "")
                row["processing_time_sec"] = f"{elapsed:.3f}"
                csv_writer.writerow(row)
                csv_file.flush()
                n_ok += 1

    finally:
        csv_file.close()

    print(f"[done] ok={n_ok} error={n_error} skipped={n_skipped} total={len(meta_rows)}")
    print(f"[done] output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
