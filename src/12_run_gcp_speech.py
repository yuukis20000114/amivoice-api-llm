#!/usr/bin/env python3
"""GCP Speech-to-Text v2 で音声認識を実行し CSV を出力する.

Phase1 ASR スクリプト.

実行例:
    uv run python src/12_run_gcp_speech.py --variant clean
    uv run python src/12_run_gcp_speech.py --variant white_snr_10
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

from dotenv import dotenv_values
from google.api_core import exceptions as gcp_exceptions
from google.cloud import speech_v2
from google.cloud.speech_v2.types import cloud_speech
from google.protobuf.json_format import MessageToDict
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from asr_csv import load_existing_keys, open_asr_csv

DEFAULT_MODEL = "latest_long"
DEFAULT_LOCATION = "global"
DEFAULT_METADATA = Path("inputs/common_voice_ja_test/metadata.csv")
DEFAULT_AUDIO_DIR = Path("inputs/audio_variants")
DEFAULT_OUTPUT_DIR = Path("outputs")
DEFAULT_RAW_BASE = Path("outputs/raw_responses/gcp_speech")


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
        help="出力 CSV パス (デフォルト: outputs/asr_gcp_speech_{variant}.csv)",
    )
    p.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help="raw response 保存先 (デフォルト: outputs/raw_responses/gcp_speech/{variant}/)",
    )
    p.add_argument(
        "--model",
        default=None,
        help="GCP STT モデル (デフォルト: env GCP_SPEECH_MODEL or latest_long)",
    )
    args = p.parse_args()
    if args.output is None:
        args.output = DEFAULT_OUTPUT_DIR / f"asr_gcp_speech_{args.variant}.csv"
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


def _load_project_id(creds_path: str) -> str:
    with Path(creds_path).open(encoding="utf-8") as f:
        return json.load(f)["project_id"]


@retry(
    retry=retry_if_exception_type(
        (
            gcp_exceptions.ServiceUnavailable,
            gcp_exceptions.DeadlineExceeded,
            gcp_exceptions.ResourceExhausted,
        ),
    ),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    reraise=True,
)
def call_gcp_speech(
    client: speech_v2.SpeechClient,
    recognizer: str,
    config: cloud_speech.RecognitionConfig,
    audio_bytes: bytes,
) -> cloud_speech.RecognizeResponse:
    request = cloud_speech.RecognizeRequest(
        recognizer=recognizer,
        config=config,
        content=audio_bytes,
    )
    return client.recognize(request=request)


def _extract_text(response: cloud_speech.RecognizeResponse) -> str:
    parts = []
    for result in response.results:
        if result.alternatives:
            parts.append(result.alternatives[0].transcript)
    return "".join(parts)


def _process_sample(  # noqa: PLR0913
    meta_row: dict[str, str],
    variant: str,
    variant_dir: Path,
    raw_dir: Path,
    model: str,
    client: speech_v2.SpeechClient,
    recognizer: str,
    config: cloud_speech.RecognitionConfig,
) -> dict[str, str]:
    sample_id = meta_row["sample_id"]
    reference_text = meta_row.get("reference_text", "")
    audio_path = variant_dir / f"{sample_id}.wav"
    raw_path = raw_dir / f"{sample_id}.json"

    row: dict[str, str] = {
        "sample_id": sample_id,
        "audio_variant": variant,
        "audio_path": str(audio_path),
        "reference_text": reference_text,
        "asr_provider": "gcp_speech",
        "asr_model": model,
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
        return row

    audio_bytes = audio_path.read_bytes()

    t0 = time.monotonic()
    try:
        response = call_gcp_speech(client, recognizer, config, audio_bytes)
    except (gcp_exceptions.GoogleAPICallError, ValueError, OSError) as e:
        row["status"] = "error"
        row["error_message"] = str(e)
        row["processing_time_sec"] = f"{time.monotonic() - t0:.3f}"
        row["raw_response_path"] = ""
        return row
    elapsed = time.monotonic() - t0

    response_dict = MessageToDict(response._pb)
    raw_path.write_text(
        json.dumps(response_dict, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    row["status"] = "ok"
    row["asr_text"] = _extract_text(response)
    row["processing_time_sec"] = f"{elapsed:.3f}"
    return row


def main() -> int:
    args = parse_args()

    creds_path = get_env("GOOGLE_APPLICATION_CREDENTIALS")
    if not creds_path or not Path(creds_path).exists():
        print(
            "GOOGLE_APPLICATION_CREDENTIALS が設定されていないか、"
            "ファイルが存在しません。",
            file=sys.stderr,
        )
        return 1

    project_id = _load_project_id(creds_path)
    location = get_env("GCP_LOCATION", DEFAULT_LOCATION)
    model = args.model or get_env("GCP_SPEECH_MODEL", DEFAULT_MODEL)

    try:
        from tqdm import tqdm  # noqa: PLC0415
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

    recognizer = f"projects/{project_id}/locations/{location}/recognizers/_"
    config = cloud_speech.RecognitionConfig(
        auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
        language_codes=["ja-JP"],
        model=model,
    )

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path
    client = speech_v2.SpeechClient()

    print(
        f"[run] variant={variant} model={model} location={location} "
        f"project={project_id}",
    )

    csv_file, csv_writer = open_asr_csv(args.output)
    n_ok = 0
    n_error = 0
    n_skipped = 0

    try:
        for meta_row in tqdm(meta_rows, desc=f"gcp_speech/{variant}"):
            if (meta_row["sample_id"], variant) in existing_keys:
                n_skipped += 1
                continue

            row = _process_sample(
                meta_row,
                variant,
                variant_dir,
                raw_dir,
                model,
                client,
                recognizer,
                config,
            )
            csv_writer.writerow(row)
            csv_file.flush()
            if row["status"] == "ok":
                n_ok += 1
            else:
                n_error += 1

    finally:
        csv_file.close()

    print(
        f"[done] ok={n_ok} error={n_error} skipped={n_skipped} total={len(meta_rows)}",
    )
    print(f"[done] output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
