#!/usr/bin/env python3
"""AWS Transcribe で音声認識を実行し CSV を出力する.

Phase1 ASR スクリプト.

実行例:
    uv run python src/13_run_aws_transcribe.py --variant clean
    uv run python src/13_run_aws_transcribe.py --variant white_snr_10
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import sys
import threading
import time
import uuid
from pathlib import Path

import boto3
import requests
from dotenv import dotenv_values

sys.path.insert(0, str(Path(__file__).resolve().parent))
from asr_csv import load_existing_keys, open_asr_csv

DEFAULT_METADATA = Path("inputs/common_voice_ja_test/metadata.csv")
DEFAULT_AUDIO_DIR = Path("inputs/audio_variants")
DEFAULT_OUTPUT_DIR = Path("outputs")
DEFAULT_RAW_BASE = Path("outputs/raw_responses/aws_transcribe")
DEFAULT_S3_PREFIX = "asr-phase1/"
POLL_INTERVAL = 5


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
        "--concurrency",
        type=int,
        default=20,
        help="同時実行ジョブ数 (デフォルト: 20, AWS制限: 100)",
    )
    args = p.parse_args()
    if args.output is None:
        args.output = DEFAULT_OUTPUT_DIR / f"asr_aws_transcribe_{args.variant}.csv"
    if args.raw_dir is None:
        args.raw_dir = DEFAULT_RAW_BASE / args.variant
    return args


def get_env(key: str, default: str | None = None) -> str | None:
    val = os.environ.get(key)
    if val:
        return val
    env_file = Path(".env")
    if env_file.exists():
        vals = dotenv_values(env_file)
        val = vals.get(key)
    return val or default


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
        "asr_provider": "aws_transcribe",
        "asr_model": "default",
        "asr_text": "",
        "status": "",
        "error_message": "",
        "processing_time_sec": "",
        "raw_response_path": "",
    }


def _upload_to_s3(
    s3_client: object,
    local_path: Path,
    bucket: str,
    s3_key: str,
) -> None:
    s3_client.upload_file(str(local_path), bucket, s3_key)


def _start_job(
    transcribe_client: object,
    job_name: str,
    media_uri: str,
) -> None:
    transcribe_client.start_transcription_job(
        TranscriptionJobName=job_name,
        LanguageCode="ja-JP",
        MediaFormat="wav",
        Media={"MediaFileUri": media_uri},
    )


def _wait_for_job(transcribe_client: object, job_name: str) -> dict:
    while True:
        resp = transcribe_client.get_transcription_job(
            TranscriptionJobName=job_name,
        )
        job = resp["TranscriptionJob"]
        status = job["TranscriptionJobStatus"]
        if status in ("COMPLETED", "FAILED"):
            return job
        time.sleep(POLL_INTERVAL)


def _download_transcript(transcript_uri: str) -> dict:
    resp = requests.get(transcript_uri, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _extract_text(transcript_json: dict) -> str:
    results = transcript_json.get("results", {})
    transcripts = results.get("transcripts", [])
    if transcripts:
        return transcripts[0].get("transcript", "")
    return ""


def _process_sample(  # noqa: PLR0913
    meta_row: dict[str, str],
    variant: str,
    variant_dir: Path,
    raw_dir: Path,
    s3_client: object,
    transcribe_client: object,
    bucket: str,
    s3_prefix: str,
) -> dict[str, str]:
    sample_id = meta_row["sample_id"]
    audio_path = variant_dir / f"{sample_id}.wav"
    raw_path = raw_dir / f"{sample_id}.json"
    row = _make_row(meta_row, variant, audio_path)

    if not audio_path.exists():
        row["status"] = "error"
        row["error_message"] = "audio file not found"
        return row

    s3_key = f"{s3_prefix}{variant}/{sample_id}.wav"
    media_uri = f"s3://{bucket}/{s3_key}"
    job_name = f"{sample_id}-{variant}-{uuid.uuid4().hex[:8]}"

    t0 = time.monotonic()
    try:
        _upload_to_s3(s3_client, audio_path, bucket, s3_key)
        _start_job(transcribe_client, job_name, media_uri)
        job = _wait_for_job(transcribe_client, job_name)
    except (
        boto3.exceptions.Boto3Error,
        requests.exceptions.RequestException,
        Exception,  # noqa: BLE001
    ) as e:
        row["status"] = "error"
        row["error_message"] = str(e)
        row["processing_time_sec"] = f"{time.monotonic() - t0:.3f}"
        return row

    status = job["TranscriptionJobStatus"]
    if status == "FAILED":
        row["status"] = "error"
        row["error_message"] = job.get("FailureReason", "unknown failure")
        row["processing_time_sec"] = f"{time.monotonic() - t0:.3f}"
        return row

    try:
        transcript_uri = job["Transcript"]["TranscriptFileUri"]
        transcript_json = _download_transcript(transcript_uri)
    except (
        requests.exceptions.RequestException,
        KeyError,
        ValueError,
    ) as e:
        row["status"] = "error"
        row["error_message"] = f"transcript download failed: {e}"
        row["processing_time_sec"] = f"{time.monotonic() - t0:.3f}"
        return row
    elapsed = time.monotonic() - t0

    raw_path.write_text(
        json.dumps(transcript_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    row["status"] = "ok"
    row["asr_text"] = _extract_text(transcript_json)
    row["processing_time_sec"] = f"{elapsed:.3f}"
    row["raw_response_path"] = str(raw_path)
    return row


def _load_aws_config() -> tuple[str, str, str, str, str] | None:
    bucket = get_env("AWS_TRANSCRIBE_S3_BUCKET")
    if not bucket:
        print("AWS_TRANSCRIBE_S3_BUCKET が設定されていません。", file=sys.stderr)
        return None

    aws_key = get_env("AWS_ACCESS_KEY_ID")
    aws_secret = get_env("AWS_SECRET_ACCESS_KEY")
    if not aws_key or not aws_secret:
        print(
            "AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY が設定されていません。",
            file=sys.stderr,
        )
        return None

    region = get_env("AWS_REGION", "ap-northeast-1")
    s3_prefix = get_env("AWS_TRANSCRIBE_S3_PREFIX", DEFAULT_S3_PREFIX)
    return aws_key, aws_secret, region, bucket, s3_prefix


_thread_local = threading.local()


def _get_thread_clients(
    aws_key: str,
    aws_secret: str,
    region: str,
) -> tuple[object, object]:
    if not hasattr(_thread_local, "s3_client"):
        session = boto3.Session(
            aws_access_key_id=aws_key,
            aws_secret_access_key=aws_secret,
            region_name=region,
        )
        _thread_local.s3_client = session.client("s3")
        _thread_local.transcribe_client = session.client("transcribe")
    return _thread_local.s3_client, _thread_local.transcribe_client


def _process_sample_threaded(  # noqa: PLR0913
    meta_row: dict[str, str],
    variant: str,
    variant_dir: Path,
    raw_dir: Path,
    aws_key: str,
    aws_secret: str,
    region: str,
    bucket: str,
    s3_prefix: str,
) -> dict[str, str]:
    s3_client, transcribe_client = _get_thread_clients(aws_key, aws_secret, region)
    return _process_sample(
        meta_row,
        variant,
        variant_dir,
        raw_dir,
        s3_client,
        transcribe_client,
        bucket,
        s3_prefix,
    )


def _run_concurrent(  # noqa: PLR0913
    pending: list[dict[str, str]],
    variant: str,
    variant_dir: Path,
    raw_dir: Path,
    aws_key: str,
    aws_secret: str,
    region: str,
    bucket: str,
    s3_prefix: str,
    concurrency: int,
    output_path: Path,
    tqdm: type,
) -> tuple[int, int]:
    csv_file, csv_writer = open_asr_csv(output_path)
    n_ok = 0
    n_error = 0

    try:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=concurrency,
        ) as executor:
            future_to_sample_id: dict[concurrent.futures.Future, str] = {}
            for meta_row in pending:
                future = executor.submit(
                    _process_sample_threaded,
                    meta_row,
                    variant,
                    variant_dir,
                    raw_dir,
                    aws_key,
                    aws_secret,
                    region,
                    bucket,
                    s3_prefix,
                )
                future_to_sample_id[future] = meta_row["sample_id"]

            with tqdm(
                total=len(future_to_sample_id),
                desc=f"aws_transcribe/{variant}",
            ) as pbar:
                for future in concurrent.futures.as_completed(
                    future_to_sample_id,
                ):
                    sample_id = future_to_sample_id[future]
                    try:
                        row = future.result()
                    except Exception as exc:  # noqa: BLE001
                        row = {
                            "sample_id": sample_id,
                            "audio_variant": variant,
                            "audio_path": "",
                            "reference_text": "",
                            "asr_provider": "aws_transcribe",
                            "asr_model": "default",
                            "asr_text": "",
                            "status": "error",
                            "error_message": f"thread exception: {exc}",
                            "processing_time_sec": "",
                            "raw_response_path": "",
                        }

                    csv_writer.writerow(row)
                    csv_file.flush()
                    if row["status"] == "ok":
                        n_ok += 1
                    else:
                        n_error += 1
                    pbar.update(1)
    finally:
        csv_file.close()

    return n_ok, n_error


def main() -> int:
    args = parse_args()

    config = _load_aws_config()
    if config is None:
        return 1
    aws_key, aws_secret, region, bucket, s3_prefix = config

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

    pending = [
        row for row in meta_rows if (row["sample_id"], variant) not in existing_keys
    ]
    print(f"[pending] {len(pending)} samples to process")

    if not pending:
        print("[done] nothing to do")
        return 0

    raw_dir: Path = args.raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)

    concurrency: int = args.concurrency
    print(
        f"[run] variant={variant} bucket={bucket} region={region} "
        f"concurrency={concurrency}",
    )

    n_ok, n_error = _run_concurrent(
        pending,
        variant,
        variant_dir,
        raw_dir,
        aws_key,
        aws_secret,
        region,
        bucket,
        s3_prefix,
        concurrency,
        args.output,
        tqdm,
    )

    print(
        f"[done] ok={n_ok} error={n_error} skipped={len(existing_keys)} "
        f"total={len(meta_rows)}",
    )
    print(f"[done] output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
