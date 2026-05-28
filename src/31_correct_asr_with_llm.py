"""AmiVoice ASR 結果に対する LLM 誤り訂正."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
from pathlib import Path

from tqdm import tqdm

from llm_utils import (
    call_llm,
    extract_json_from_response,
    format_low_confidence_tokens,
    load_amivoice_confidence,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

CORRECTION_CSV_FIELDS = [
    "sample_id",
    "audio_variant",
    "reference_text",
    "asr_text_original",
    "corrector_model",
    "corrected_text",
    "correction_changed",
    "low_confidence_tokens",
    "prompt_template_version",
    "processing_time_sec",
    "status",
    "error_message",
    "raw_response_path",
]

PROMPT_TEMPLATE_V1 = """\
あなたは日本語音声認識(ASR)の誤り訂正の専門家です。

## タスク
以下のASR認識結果テキストを読み、明らかな誤りのみを修正してください。
確信が持てない箇所は絶対に変更しないでください。

## ASR認識結果
{asr_text}

## 低信頼度トークン（誤りの可能性が高い箇所）
{low_confidence_tokens_formatted}

## 修正ルール
1. 音声として自然な日本語になるよう修正する
2. 文意が通らない箇所を優先的に修正する
3. 低信頼度トークンを重点的に確認する
4. 表記の好みの差（漢数字vsアラビア数字、ひらがなvsカタカナ等）は修正しない
5. 確信度が低い修正は行わない（元のテキストを保持する）
6. テキストが正しいと判断した場合は、そのまま返す

## 修正例
入力: "光の白い制服を気にかかりますすぐな道をまっすぐにいきました"
低信頼度: [{{"token": "制", "confidence": 0.65}}, {{"token": "かかり", "confidence": 0.71}}]
出力: {{"corrected_text": "光の城の製服を着にかかる真っ直ぐな道をまっすぐにいきました", "corrections": [{{"original": "白い制", "corrected": "城の製", "reason": "同音異義語の取り違え"}}]}}

入力: "挑戦という言葉を忘れた"
低信頼度: []
出力: {{"corrected_text": "挑戦という言葉を忘れた", "corrections": []}}

## 出力形式
以下のJSON形式のみで出力してください。説明文は不要です:
{{"corrected_text": "修正後のテキスト", "corrections": [{{"original": "元", "corrected": "修正後", "reason": "理由"}}]}}\
"""

SYSTEM_MESSAGE_V1 = "JSONのみ出力してください。説明や思考は不要です。"

PROMPT_TEMPLATE_V2 = """\
<task>
以下のASR（音声認識）結果テキストの誤認識を修正してください。
</task>

<asr_text>
{asr_text}
</asr_text>

<low_confidence_tokens>
{low_confidence_tokens_formatted}
</low_confidence_tokens>

<guidelines>
原則: 迷ったら変更しない。誤った修正は、修正しないことより悪い。

修正すべきもの（以下のすべてを満たす場合のみ）:
- 音声的に説明できる取り違えである（同音異義語、類音語、文の区切りずれ）
- 正しい語が文脈から高い確信をもって特定できる
- 修正により文全体の意味が改善される

修正してはいけないもの:
- 文法的に正しく意味が通る文（正解と異なっていても変更しない）
- 句読点の有無や位置の違い
- 表記スタイルの差（漢数字/アラビア数字、ひらがな/カタカナ、送り仮名等）
- 正しい語の候補が複数あり絞り込めない箇所
- 元テキストが大幅に崩壊しており正しい文を推測できない箇所

信頼度スコアは参考情報です。高信頼度(1.0)でも誤りの場合があり、低信頼度でも正しい場合があります。
</guidelines>

<examples>
入力: "レンガ制の倉庫も海にかかります。すぐな橋も全てが瓦礫とかしていた"
低信頼度: [{{"token": "制", "confidence": 0.82}}, {{"token": "ます", "confidence": 0.65}}, {{"token": "か", "confidence": 0.70}}]
出力: {{"corrected_text": "レンガ製の倉庫も海にかかる真っ直ぐな橋も全てが瓦礫と化していた"}}

入力: "疲労回復には十分な睡眠が必要だ。"
低信頼度: [{{"token": "に", "confidence": 0.70}}]
出力: {{"corrected_text": "疲労回復には十分な睡眠が必要だ。"}}
</examples>

JSON形式のみで出力: {{"corrected_text": "修正後のテキスト"}}\
"""

SYSTEM_MESSAGE_V2 = (
    "あなたは日本語ASR誤り訂正システムです。"
    "入力されたASRテキストの誤認識を修正し、JSON形式で返してください。"
    "説明や思考は不要です。"
)

_PROMPT_TEMPLATES = {
    "v1": PROMPT_TEMPLATE_V1,
    "v2": PROMPT_TEMPLATE_V2,
}

_SYSTEM_MESSAGES = {
    "v1": SYSTEM_MESSAGE_V1,
    "v2": SYSTEM_MESSAGE_V2,
}


def _build_prompt(
    asr_text: str,
    low_confidence_tokens: list[dict],
    version: str = "v1",
) -> str:
    template = _PROMPT_TEMPLATES.get(version)
    if template is None:
        msg = f"Unknown prompt version: {version}"
        raise ValueError(msg)
    return template.format(
        asr_text=asr_text,
        low_confidence_tokens_formatted=format_low_confidence_tokens(
            low_confidence_tokens,
        ),
    )


def _get_system_message(version: str = "v1") -> str:
    return _SYSTEM_MESSAGES.get(version, SYSTEM_MESSAGE_V1)


def _load_existing_sample_ids(csv_path: Path) -> set[str]:
    if not csv_path.exists():
        return set()
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {row["sample_id"] for row in reader if row.get("sample_id")}


def _open_correction_csv(csv_path: Path) -> tuple[object, csv.DictWriter]:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    f = csv_path.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=CORRECTION_CSV_FIELDS)
    if not file_exists:
        writer.writeheader()
    return f, writer


def main() -> None:  # noqa: PLR0915
    parser = argparse.ArgumentParser(
        description="AmiVoice ASR 結果を LLM で訂正する",
    )
    parser.add_argument(
        "--asr-csv",
        type=Path,
        default=Path("outputs/asr_amivoice_clean.csv"),
    )
    parser.add_argument(
        "--raw-response-dir",
        type=Path,
        default=Path("outputs/raw_responses/amivoice/clean"),
    )
    parser.add_argument("--model", default="claude-sonnet-4-6")
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.8,
    )
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="ランダムサンプリング用シード",
    )
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/llm_correction"),
    )
    parser.add_argument("--prompt-version", default="v1")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="処理件数の上限（デバッグ用）",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="ランダムサンプリング件数（--seed で再現可能）",
    )
    args = parser.parse_args()

    output_csv = args.output_dir / "corrected_amivoice.csv"
    raw_out_dir = args.output_dir / "correction_raw"
    raw_out_dir.mkdir(parents=True, exist_ok=True)

    log.info("ASR CSV 読み込み: %s", args.asr_csv)
    with args.asr_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        asr_rows = list(reader)
    log.info("全行数: %d", len(asr_rows))

    existing_ids = _load_existing_sample_ids(output_csv)
    log.info("既存処理済み: %d 件", len(existing_ids))

    csv_file, writer = _open_correction_csv(output_csv)

    stats = {"ok": 0, "skipped": 0, "error": 0, "changed": 0, "unchanged": 0}

    try:
        rows_to_process = [r for r in asr_rows if r["status"] == "ok"]
        if args.sample and args.sample < len(rows_to_process):
            rng = random.Random(args.seed)
            rows_to_process = rng.sample(rows_to_process, args.sample)
            log.info("ランダムサンプリング: %d 件", len(rows_to_process))
        elif args.limit:
            rows_to_process = rows_to_process[: args.limit]

        for row in tqdm(rows_to_process, desc="LLM correction"):
            sample_id = row["sample_id"]

            if sample_id in existing_ids:
                continue

            if row["status"] != "ok":
                out = {
                    "sample_id": sample_id,
                    "audio_variant": row["audio_variant"],
                    "reference_text": row["reference_text"],
                    "asr_text_original": row.get("asr_text", ""),
                    "corrector_model": args.model,
                    "corrected_text": "",
                    "correction_changed": "false",
                    "low_confidence_tokens": "[]",
                    "prompt_template_version": args.prompt_version,
                    "processing_time_sec": 0,
                    "status": "skipped",
                    "error_message": f"ASR status={row['status']}",
                    "raw_response_path": "",
                }
                writer.writerow(out)
                csv_file.flush()
                stats["skipped"] += 1
                continue

            asr_text = row.get("asr_text", "")
            raw_path = args.raw_response_dir / f"{sample_id}.json"

            low_conf = []
            if raw_path.exists():
                low_conf = load_amivoice_confidence(
                    raw_path,
                    args.confidence_threshold,
                )

            prompt = _build_prompt(asr_text, low_conf, args.prompt_version)
            sys_msg = _get_system_message(args.prompt_version)

            try:
                resp = call_llm(
                    prompt,
                    model=args.model,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    system_message=sys_msg,
                )
                elapsed = resp["elapsed_sec"]

                parsed = extract_json_from_response(resp["content"])
                if parsed and "corrected_text" in parsed:
                    corrected = parsed["corrected_text"]
                else:
                    corrected = asr_text
                    log.warning(
                        "%s: JSON抽出失敗、ASR元テキストを使用",
                        sample_id,
                    )

                changed = corrected != asr_text
                status = "ok"
                error_msg = ""

                if changed:
                    stats["changed"] += 1
                else:
                    stats["unchanged"] += 1
                stats["ok"] += 1

            except Exception as e:  # noqa: BLE001
                corrected = asr_text
                changed = False
                elapsed = 0
                status = "error"
                error_msg = str(e)
                resp = {"content": "", "model": args.model}
                stats["error"] += 1
                log.warning("Error on %s: %s", sample_id, e)

            raw_out_path = raw_out_dir / f"{sample_id}.json"
            with raw_out_path.open("w", encoding="utf-8") as rf:
                json.dump(resp, rf, ensure_ascii=False, indent=2)

            out = {
                "sample_id": sample_id,
                "audio_variant": row["audio_variant"],
                "reference_text": row["reference_text"],
                "asr_text_original": asr_text,
                "corrector_model": args.model,
                "corrected_text": corrected,
                "correction_changed": str(changed).lower(),
                "low_confidence_tokens": json.dumps(
                    low_conf,
                    ensure_ascii=False,
                ),
                "prompt_template_version": args.prompt_version,
                "processing_time_sec": round(elapsed, 3),
                "status": status,
                "error_message": error_msg,
                "raw_response_path": str(raw_out_path),
            }
            writer.writerow(out)
            csv_file.flush()

    finally:
        csv_file.close()

    log.info("=== 訂正完了 ===")
    log.info(
        "OK: %d (changed: %d, unchanged: %d), skipped: %d, error: %d",
        stats["ok"],
        stats["changed"],
        stats["unchanged"],
        stats["skipped"],
        stats["error"],
    )


if __name__ == "__main__":
    main()
