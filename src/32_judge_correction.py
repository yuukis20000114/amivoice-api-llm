"""LLM-as-a-Judge: LASER 方式 LLM-CER + Intent/Entity 評価.

修正前（ASR元出力）と修正後（LLM訂正出力）の両方を評価し、
改善度を直接比較可能にする。

未変更サンプル（correction_changed=false）は asr_original のみ評価し、
llm_corrected には同一結果をコピーする（テキストが同一のため）。
asyncio による並列 API 呼び出しで高速化。
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
from pathlib import Path

from tqdm import tqdm

from llm_utils import call_llm_async, extract_json_from_response
from text_normalization import normalize_ja_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

JUDGE_CSV_FIELDS = [
    "sample_id",
    "audio_variant",
    "judge_model",
    "target_type",
    "target_text",
    "reference_text",
    "llm_cer",
    "major_penalty_count",
    "minor_penalty_count",
    "major_penalties_json",
    "minor_penalties_json",
    "no_penalties_json",
    "intent_score",
    "intent_reason",
    "entities_in_ref",
    "entities_preserved",
    "entity_preservation",
    "processing_time_sec",
    "status",
    "error_message",
    "raw_response_path",
]

LLM_CER_PROMPT = """\
あなたは日本語ASR（音声認識）出力の品質を評価する専門家です。

## タスク
正解テキストと評価対象テキストを比較し、各差異を以下の3段階で分類してください。

## 正解テキスト
{reference_text}

## 評価対象テキスト
{target_text}

## ペナルティ分類ルール

### No-Penalty（0.0点）— 意味を変えない表記差
- 句読点の有無（「忘れた」vs「忘れた。」）
- ひらがな/カタカナの差（「ふぇあ」vs「フェア」）
- 漢数字/アラビア数字の差（「二十」vs「20」）
- 送り仮名の差（「行なう」vs「行う」）
- 全角/半角の差
- 音訳・外来語の表記差（「コンピューター」vs「コンピュータ」）

### Minor-Penalty（0.5点）— 軽微な文法的差異
- 意味を保持する助詞の差（「を」vs「は」で文意が変わらない場合）
- 活用形の軽微な差（「行った」vs「行きました」）
- フィラー・感嘆詞の有無（「えーと」等）

### Major-Penalty（1.0点）— 意味を変える差異
- 同音異義語の取り違え（「城」vs「白」、「製」vs「制」）
- 内容語の置換・欠落・挿入
- 否定の反転（「ありません」vs「あります」）
- 主語・目的語の取り違え
- 文の区切り位置の誤りによる意味変化

## 出力例

正解: "光の城の製服を着にかかる真っ直ぐな道"
評価: "光の白い制服を気にかかりますすぐな道"

分析:
- 「城の」→「白い」: 同音異義語の取り違え → Major-Penalty (1.0)
- 「製」→「制」: 同音異義語 → Major-Penalty (1.0)
- 「着に」→「気に」: 内容語の置換 → Major-Penalty (1.0)
- 「真っ直ぐ」→「ますすぐ」: 文区切り誤り → Major-Penalty (1.0)

Major: 4件 (4.0), Minor: 0件 (0.0), Reference文字数: 16
LLM_CER = 4.0 / 16 = 0.250

## 出力形式（JSONのみ）
{{"major_penalties": [{{"ref": "元", "hyp": "評価側", "reason": "理由"}}], "minor_penalties": [], "no_penalties": [], "major_count": 0, "minor_count": 0, "ref_char_count": 0, "llm_cer": 0.0}}\
"""

INTENT_ENTITY_PROMPT = """\
以下の正解テキストと評価対象テキストを比較し、2つの指標を評価してください。

## 正解テキスト
{reference_text}

## 評価対象テキスト
{target_text}

## 評価指標

### Intent Score（意図保持）
文の核心メッセージ（誰が・何を・どうした）が維持されているか。
- 1: 核心メッセージが維持されている（軽微な表記差・同義語の使用は許容）
- 0: 主客転倒、否定反転、動作の変更など、核心的な意味が変わっている

### Entity Preservation（エンティティ保持率）
正解テキスト中の固有名詞・数値・日付・地名・人名が正しく保持されているか。
- 保持率 = 正しく保持されたエンティティ数 / 正解中の全エンティティ数
- エンティティがない場合は1.0

## 出力形式（JSONのみ）
{{"intent_score": 0, "intent_reason": "判定理由", "entities_in_ref": [], "entities_preserved": [], "entity_preservation": 1.0}}\
"""


def parse_llm_cer_response(parsed: dict | None) -> dict:
    """LLM-CER レスポンスをパースし、正規化した辞書を返す."""
    if parsed is None:
        return {
            "llm_cer": "",
            "major_penalty_count": "",
            "minor_penalty_count": "",
            "major_penalties_json": "[]",
            "minor_penalties_json": "[]",
            "no_penalties_json": "[]",
        }

    major_count = parsed.get("major_count", 0)
    minor_count = parsed.get("minor_count", 0)

    try:
        major_count = int(major_count)
    except (TypeError, ValueError):
        major_count = len(parsed.get("major_penalties", []))

    try:
        minor_count = int(minor_count)
    except (TypeError, ValueError):
        minor_count = len(parsed.get("minor_penalties", []))

    llm_cer = parsed.get("llm_cer")
    if llm_cer is None:
        ref_count = parsed.get("ref_char_count", 0)
        try:
            ref_count = int(ref_count)
        except (TypeError, ValueError):
            ref_count = 0
        total_penalty = major_count * 1.0 + minor_count * 0.5
        llm_cer = total_penalty / ref_count if ref_count > 0 else 0.0

    try:
        llm_cer = float(llm_cer)
    except (TypeError, ValueError):
        llm_cer = 0.0

    return {
        "llm_cer": round(llm_cer, 6),
        "major_penalty_count": major_count,
        "minor_penalty_count": minor_count,
        "major_penalties_json": json.dumps(
            parsed.get("major_penalties", []),
            ensure_ascii=False,
        ),
        "minor_penalties_json": json.dumps(
            parsed.get("minor_penalties", []),
            ensure_ascii=False,
        ),
        "no_penalties_json": json.dumps(
            parsed.get("no_penalties", []),
            ensure_ascii=False,
        ),
    }


def parse_intent_entity_response(parsed: dict | None) -> dict:
    """Intent/Entity レスポンスをパースし、正規化した辞書を返す."""
    if parsed is None:
        return {
            "intent_score": "",
            "intent_reason": "",
            "entities_in_ref": "[]",
            "entities_preserved": "[]",
            "entity_preservation": "",
        }

    intent = parsed.get("intent_score", "")
    try:
        intent = int(intent)
        intent = 1 if intent >= 1 else 0
    except (TypeError, ValueError):
        intent = ""

    ep = parsed.get("entity_preservation", "")
    try:
        ep = float(ep)
        ep = max(0.0, min(1.0, ep))
        ep = round(ep, 4)
    except (TypeError, ValueError):
        ep = ""

    return {
        "intent_score": intent,
        "intent_reason": parsed.get("intent_reason", ""),
        "entities_in_ref": json.dumps(
            parsed.get("entities_in_ref", []),
            ensure_ascii=False,
        ),
        "entities_preserved": json.dumps(
            parsed.get("entities_preserved", []),
            ensure_ascii=False,
        ),
        "entity_preservation": ep,
    }


def _load_existing_keys(csv_path: Path) -> set[tuple[str, str]]:
    """既存 CSV から (sample_id, target_type) のペアを返す."""
    if not csv_path.exists():
        return set()
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {
            (row["sample_id"], row["target_type"])
            for row in reader
            if row.get("sample_id") and row.get("target_type")
        }


def _open_judge_csv(csv_path: Path) -> tuple[object, csv.DictWriter]:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    f = csv_path.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=JUDGE_CSV_FIELDS)
    if not file_exists:
        writer.writeheader()
    return f, writer


async def _judge_one_async(  # noqa: PLR0913
    reference_text: str,
    target_text: str,
    model: str,
    temperature: float,
    max_tokens: int,
    semaphore: asyncio.Semaphore,
) -> tuple[dict, dict, list[dict]]:
    """1件の Judge 評価（LLM-CER + Intent/Entity）を非同期で実行する."""
    raw_responses = []

    async with semaphore:
        resp_cer = await call_llm_async(
            LLM_CER_PROMPT.format(
                reference_text=reference_text,
                target_text=target_text,
            ),
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            system_message="JSONのみ出力してください。説明や思考は不要です。",
        )
    raw_responses.append({"type": "llm_cer", **resp_cer})
    cer_parsed = extract_json_from_response(resp_cer["content"])
    cer_result = parse_llm_cer_response(cer_parsed)

    async with semaphore:
        resp_ie = await call_llm_async(
            INTENT_ENTITY_PROMPT.format(
                reference_text=reference_text,
                target_text=target_text,
            ),
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            system_message="JSONのみ出力してください。説明や思考は不要です。",
        )
    raw_responses.append({"type": "intent_entity", **resp_ie})
    ie_parsed = extract_json_from_response(resp_ie["content"])
    ie_result = parse_intent_entity_response(ie_parsed)

    return cer_result, ie_result, raw_responses


def _is_perfect_match(
    reference_text: str,
    asr_text: str,
    correction_changed: str,
) -> bool:
    """CER=0 かつ未変更のサンプルを判定する."""
    ref_norm = normalize_ja_text(reference_text, "ja_surface_v1")
    asr_norm = normalize_ja_text(asr_text, "ja_surface_v1")
    return ref_norm == asr_norm and correction_changed == "false"


def _perfect_match_row(  # noqa: PLR0913
    sample_id: str,
    audio_variant: str,
    judge_model: str,
    target_type: str,
    target_text: str,
    reference_text: str,
) -> dict:
    return {
        "sample_id": sample_id,
        "audio_variant": audio_variant,
        "judge_model": judge_model,
        "target_type": target_type,
        "target_text": target_text,
        "reference_text": reference_text,
        "llm_cer": 0.0,
        "major_penalty_count": 0,
        "minor_penalty_count": 0,
        "major_penalties_json": "[]",
        "minor_penalties_json": "[]",
        "no_penalties_json": "[]",
        "intent_score": 1,
        "intent_reason": "CER=0, 完全一致",
        "entities_in_ref": "[]",
        "entities_preserved": "[]",
        "entity_preservation": 1.0,
        "processing_time_sec": 0,
        "status": "ok",
        "error_message": "",
        "raw_response_path": "",
    }


def _error_row(  # noqa: PLR0913
    sample_id: str,
    audio_variant: str,
    judge_model: str,
    target_type: str,
    target_text: str,
    reference_text: str,
    error: str,
) -> dict:
    return {
        "sample_id": sample_id,
        "audio_variant": audio_variant,
        "judge_model": judge_model,
        "target_type": target_type,
        "target_text": target_text,
        "reference_text": reference_text,
        "llm_cer": "",
        "major_penalty_count": "",
        "minor_penalty_count": "",
        "major_penalties_json": "[]",
        "minor_penalties_json": "[]",
        "no_penalties_json": "[]",
        "intent_score": "",
        "intent_reason": "",
        "entities_in_ref": "[]",
        "entities_preserved": "[]",
        "entity_preservation": "",
        "processing_time_sec": 0,
        "status": "error",
        "error_message": error,
        "raw_response_path": "",
    }


async def _process_one_sample(  # noqa: C901, PLR0912, PLR0913, PLR0915
    row: dict,
    args: argparse.Namespace,
    existing_keys: set[tuple[str, str]],
    semaphore: asyncio.Semaphore,
    csv_lock: asyncio.Lock,
    writer: csv.DictWriter,
    csv_file: object,
    raw_out_dir: Path,
    stats: dict,
    pbar: tqdm,
) -> None:
    """1サンプル（修正前後）の Judge 評価を処理する."""
    sample_id = row["sample_id"]
    audio_variant = row["audio_variant"]
    reference_text = row["reference_text"]
    asr_text = row["asr_text_original"]
    corrected_text = row["corrected_text"]
    correction_changed = row.get("correction_changed", "false")

    if row.get("status") == "skipped":
        pbar.update(1)
        return

    is_perfect = _is_perfect_match(reference_text, asr_text, correction_changed)
    is_unchanged = correction_changed.lower() != "true"

    if is_unchanged:
        both_existing = (sample_id, "asr_original") in existing_keys and (
            sample_id,
            "llm_corrected",
        ) in existing_keys
        if both_existing:
            stats["skipped_existing"] += 2
            pbar.update(1)
            return

    if is_perfect:
        rows_to_write = []
        for tt in ("asr_original", "llm_corrected"):
            if (sample_id, tt) not in existing_keys:
                text = asr_text if tt == "asr_original" else corrected_text
                rows_to_write.append(
                    _perfect_match_row(
                        sample_id,
                        audio_variant,
                        args.judge_model,
                        tt,
                        text,
                        reference_text,
                    ),
                )
                stats["skipped_perfect"] += 1
        if rows_to_write:
            async with csv_lock:
                for r in rows_to_write:
                    writer.writerow(r)
                csv_file.flush()
        pbar.update(1)
        return

    if is_unchanged:
        asr_already = (sample_id, "asr_original") in existing_keys
        if not asr_already:
            try:
                cer_result, ie_result, raw_responses = await _judge_one_async(
                    reference_text,
                    asr_text,
                    model=args.judge_model,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    semaphore=semaphore,
                )
                elapsed = sum(r.get("elapsed_sec", 0) for r in raw_responses)

                raw_path = raw_out_dir / f"{sample_id}_asr_original.json"
                raw_path.write_text(
                    json.dumps(raw_responses, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

                asr_row = {
                    "sample_id": sample_id,
                    "audio_variant": audio_variant,
                    "judge_model": args.judge_model,
                    "target_type": "asr_original",
                    "target_text": asr_text,
                    "reference_text": reference_text,
                    **cer_result,
                    **ie_result,
                    "processing_time_sec": round(elapsed, 3),
                    "status": "ok",
                    "error_message": "",
                    "raw_response_path": str(raw_path),
                }
                copy_row = {**asr_row, "target_type": "llm_corrected"}

                async with csv_lock:
                    writer.writerow(asr_row)
                    writer.writerow(copy_row)
                    csv_file.flush()
                stats["ok"] += 1
                stats["copied_unchanged"] += 1

            except Exception as e:  # noqa: BLE001
                async with csv_lock:
                    for tt in ("asr_original", "llm_corrected"):
                        writer.writerow(
                            _error_row(
                                sample_id,
                                audio_variant,
                                args.judge_model,
                                tt,
                                asr_text,
                                reference_text,
                                str(e),
                            ),
                        )
                    csv_file.flush()
                stats["error"] += 2
                log.warning("Error on %s (unchanged): %s", sample_id, e)
        elif (sample_id, "llm_corrected") not in existing_keys:
            stats["copied_unchanged"] += 1
        pbar.update(1)
        return

    for target_type, target_text in [
        ("asr_original", asr_text),
        ("llm_corrected", corrected_text),
    ]:
        if (sample_id, target_type) in existing_keys:
            stats["skipped_existing"] += 1
            continue

        try:
            cer_result, ie_result, raw_responses = await _judge_one_async(
                reference_text,
                target_text,
                model=args.judge_model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                semaphore=semaphore,
            )
            elapsed = sum(r.get("elapsed_sec", 0) for r in raw_responses)

            raw_path = raw_out_dir / f"{sample_id}_{target_type}.json"
            raw_path.write_text(
                json.dumps(raw_responses, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            out = {
                "sample_id": sample_id,
                "audio_variant": audio_variant,
                "judge_model": args.judge_model,
                "target_type": target_type,
                "target_text": target_text,
                "reference_text": reference_text,
                **cer_result,
                **ie_result,
                "processing_time_sec": round(elapsed, 3),
                "status": "ok",
                "error_message": "",
                "raw_response_path": str(raw_path),
            }
            async with csv_lock:
                writer.writerow(out)
                csv_file.flush()
            stats["ok"] += 1

        except Exception as e:  # noqa: BLE001
            async with csv_lock:
                writer.writerow(
                    _error_row(
                        sample_id,
                        audio_variant,
                        args.judge_model,
                        target_type,
                        target_text,
                        reference_text,
                        str(e),
                    ),
                )
                csv_file.flush()
            stats["error"] += 1
            log.warning("Error on %s/%s: %s", sample_id, target_type, e)

    pbar.update(1)


async def async_main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM-as-a-Judge: 修正前後の意味的評価",
    )
    parser.add_argument(
        "--corrected-csv",
        type=Path,
        default=Path("outputs/llm_correction/corrected_amivoice.csv"),
    )
    parser.add_argument("--judge-model", default="claude-sonnet-4-6")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/llm_correction"),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="処理件数の上限（デバッグ用）",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="並列 API 呼び出し数（デフォルト: 10）",
    )
    args = parser.parse_args()

    output_csv = args.output_dir / "judge_scores.csv"
    raw_out_dir = args.output_dir / "judge_raw"
    raw_out_dir.mkdir(parents=True, exist_ok=True)

    log.info("訂正 CSV 読み込み: %s", args.corrected_csv)
    with args.corrected_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    log.info("全行数: %d", len(rows))

    existing_keys = _load_existing_keys(output_csv)
    log.info("既存処理済みキー: %d 件", len(existing_keys))

    csv_file, writer = _open_judge_csv(output_csv)

    stats = {
        "ok": 0,
        "skipped_perfect": 0,
        "skipped_existing": 0,
        "copied_unchanged": 0,
        "error": 0,
    }

    semaphore = asyncio.Semaphore(args.concurrency)
    csv_lock = asyncio.Lock()

    rows_to_process = rows
    if args.limit:
        rows_to_process = rows[: args.limit]

    pbar = tqdm(total=len(rows_to_process), desc="Judge evaluation")

    try:
        tasks = [
            _process_one_sample(
                row,
                args,
                existing_keys,
                semaphore,
                csv_lock,
                writer,
                csv_file,
                raw_out_dir,
                stats,
                pbar,
            )
            for row in rows_to_process
        ]
        await asyncio.gather(*tasks)
    finally:
        pbar.close()
        csv_file.close()

    log.info("=== Judge 評価完了 ===")
    log.info(
        "OK: %d, copied_unchanged: %d, perfect_skip: %d, existing_skip: %d, error: %d",
        stats["ok"],
        stats["copied_unchanged"],
        stats["skipped_perfect"],
        stats["skipped_existing"],
        stats["error"],
    )


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
