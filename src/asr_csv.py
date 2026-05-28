"""ASR スクリプト共通の CSV ヘルパ."""

from __future__ import annotations

import csv
from pathlib import Path

ASR_CSV_FIELDS = [
    "sample_id",
    "audio_variant",
    "audio_path",
    "reference_text",
    "asr_provider",
    "asr_model",
    "asr_text",
    "status",
    "error_message",
    "processing_time_sec",
    "raw_response_path",
]


def load_existing_keys(csv_path: Path) -> set[tuple[str, str]]:
    """既存 CSV から (sample_id, audio_variant) のペアを返す."""
    if not csv_path.exists():
        return set()
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {
            (row["sample_id"], row["audio_variant"])
            for row in reader
            if row.get("sample_id") and row.get("audio_variant")
        }


def open_asr_csv(csv_path: Path) -> tuple[object, csv.DictWriter]:
    """追記モードで CSV をオープンし (file, DictWriter) を返す.

    ファイルが無ければヘッダを書き込む.
    呼び出し側で file.close() すること.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    f = csv_path.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=ASR_CSV_FIELDS)
    if not file_exists:
        writer.writeheader()
    return f, writer
