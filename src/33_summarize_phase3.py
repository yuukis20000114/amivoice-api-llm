"""Phase3 統合サマリ: CER 再評価 + Judge 集計（修正前後の比較）."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from scipy import stats as sp_stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

from text_normalization import normalize_ja_text, tokenize_ja_words

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

PHASE3_SUMMARY_FIELDS = [
    "corrector_model",
    "n_total",
    "n_ok",
    "n_skipped",
    "n_corrected",
    "n_unchanged",
    "n_improved_cer",
    "n_degraded_cer",
    "n_equal_cer",
    "cer_before",
    "cer_after",
    "cer_relative_improvement_pct",
    "wer_before",
    "wer_after",
    "wer_relative_improvement_pct",
    "llm_cer_before",
    "llm_cer_after",
    "llm_cer_relative_improvement_pct",
    "intent_score_mean_before",
    "intent_score_mean_after",
    "entity_preservation_mean_before",
    "entity_preservation_mean_after",
    "spearman_cer_vs_llm_cer",
    "spearman_pvalue",
]


def _build_corrected_eval_csv(
    corrected_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """訂正後テキストを Phase1 ASR CSV 形式に変換して出力する."""
    from asr_csv import ASR_CSV_FIELDS  # noqa: PLC0415

    rows = []
    for _, row in corrected_df.iterrows():
        if row["status"] != "ok":
            continue
        rows.append(
            {
                "sample_id": row["sample_id"],
                "audio_variant": row["audio_variant"],
                "audio_path": "",
                "reference_text": row["reference_text"],
                "asr_provider": "llm_corrected",
                "asr_model": row["corrector_model"],
                "asr_text": row["corrected_text"],
                "status": "ok",
                "error_message": "",
                "processing_time_sec": row["processing_time_sec"],
                "raw_response_path": row.get("raw_response_path", ""),
            },
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ASR_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    log.info("出力: %s (%d 行)", output_path, len(rows))


def _evaluate_cer_wer(eval_csv: Path) -> pd.DataFrame:
    """21_evaluate_asr.py の内部関数を使って CER/WER を評価する."""
    import jiwer  # noqa: PLC0415

    asr_df = pd.read_csv(eval_csv)
    ok_df = asr_df[asr_df["status"] == "ok"].copy()

    if ok_df.empty:
        return pd.DataFrame()

    profile = "ja_surface_v1"
    split_mode = "C"

    refs_raw = ok_df["reference_text"].fillna("").tolist()
    hyps_raw = ok_df["asr_text"].fillna("").tolist()
    refs_norm = [normalize_ja_text(t, profile) for t in refs_raw]
    hyps_norm = [normalize_ja_text(t, profile) for t in hyps_raw]

    refs_tokens = [tokenize_ja_words(t, split_mode) for t in refs_norm]
    hyps_tokens = [tokenize_ja_words(t, split_mode) for t in hyps_norm]
    refs_words_str = [" ".join(toks) for toks in refs_tokens]
    hyps_words_str = [" ".join(toks) for toks in hyps_tokens]

    eval_indices = [i for i, r in enumerate(refs_norm) if len(r) > 0]

    results = []
    if eval_indices:
        cer_results = jiwer.process_characters(
            [refs_norm[i] for i in eval_indices],
            [hyps_norm[i] for i in eval_indices],
        )
        wer_results = jiwer.process_words(
            [refs_words_str[i] for i in eval_indices],
            [hyps_words_str[i] for i in eval_indices],
        )

        eval_pos = 0
        for idx in range(len(ok_df)):
            row = ok_df.iloc[idx]
            rec = {"sample_id": row["sample_id"]}

            if idx in set(range(len(refs_norm))) - set(eval_indices):
                rec["cer"] = None
                rec["wer"] = None
            else:
                cer_chunks = cer_results.alignments[eval_pos]
                wer_chunks = wer_results.alignments[eval_pos]

                def _sid(chunks):
                    s = sum(
                        c.ref_end_idx - c.ref_start_idx
                        for c in chunks
                        if c.type == "substitute"
                    )
                    i = sum(
                        c.hyp_end_idx - c.hyp_start_idx
                        for c in chunks
                        if c.type == "insert"
                    )
                    d = sum(
                        c.ref_end_idx - c.ref_start_idx
                        for c in chunks
                        if c.type == "delete"
                    )
                    hits = sum(
                        c.ref_end_idx - c.ref_start_idx
                        for c in chunks
                        if c.type == "equal"
                    )
                    n = hits + s + d
                    return s, i, d, n

                cs, ci, cd, cn = _sid(cer_chunks)
                ws, wi, wd, wn = _sid(wer_chunks)
                rec["cer_s"] = cs
                rec["cer_i"] = ci
                rec["cer_d"] = cd
                rec["cer_n"] = cn
                rec["cer"] = (cs + ci + cd) / cn if cn > 0 else 0.0
                rec["wer_s"] = ws
                rec["wer_i"] = wi
                rec["wer_d"] = wd
                rec["wer_n"] = wn
                rec["wer"] = (ws + wi + wd) / wn if wn > 0 else 0.0
                eval_pos += 1

            results.append(rec)

    return pd.DataFrame(results)


def _load_baseline_cer(
    baseline_path: Path,
    provider: str = "amivoice",
) -> pd.DataFrame:
    """Phase2 の utterance-level 結果から baseline CER を取得する."""
    df = pd.read_csv(baseline_path)
    mask = (
        (df["asr_provider"] == provider)
        & (df["audio_variant"] == "clean")
        & (df["status"] == "ok")
    )
    return df[mask][["sample_id", "cer", "wer"]].copy()


def _compute_judge_stats(judge_df: pd.DataFrame, target_type: str) -> dict:
    """target_type でフィルタして Judge 指標の集計を返す."""
    sub = judge_df[
        (judge_df["target_type"] == target_type) & (judge_df["status"] == "ok")
    ].copy()

    if sub.empty:
        return {
            "llm_cer_mean": None,
            "intent_score_mean": None,
            "entity_preservation_mean": None,
        }

    llm_cer = pd.to_numeric(sub["llm_cer"], errors="coerce")
    intent = pd.to_numeric(sub["intent_score"], errors="coerce")
    entity = pd.to_numeric(sub["entity_preservation"], errors="coerce")

    return {
        "llm_cer_mean": round(llm_cer.mean(), 6) if llm_cer.notna().any() else None,
        "intent_score_mean": round(intent.mean(), 4) if intent.notna().any() else None,
        "entity_preservation_mean": (
            round(entity.mean(), 4) if entity.notna().any() else None
        ),
    }


def _relative_improvement(before: float | None, after: float | None) -> str:
    if before is None or after is None or before == 0:
        return ""
    return str(round((before - after) / before * 100, 2))


def main() -> None:  # noqa: PLR0915
    parser = argparse.ArgumentParser(
        description="Phase3 統合サマリ: CER再評価 + Judge集計",
    )
    parser.add_argument(
        "--corrected-csv",
        type=Path,
        default=Path("outputs/llm_correction/corrected_amivoice.csv"),
    )
    parser.add_argument(
        "--judge-csv",
        type=Path,
        default=Path("outputs/llm_correction/judge_scores.csv"),
    )
    parser.add_argument(
        "--baseline-eval",
        type=Path,
        default=Path("outputs/evaluations/asr_eval_utterances.csv"),
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("inputs/common_voice_ja_test/metadata.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/llm_correction"),
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("訂正 CSV 読み込み: %s", args.corrected_csv)
    corrected_df = pd.read_csv(args.corrected_csv)
    ok_df = corrected_df[corrected_df["status"] == "ok"]
    corrector_model = ok_df["corrector_model"].iloc[0] if len(ok_df) > 0 else ""

    corrected_eval_path = output_dir / "corrected_for_eval.csv"
    _build_corrected_eval_csv(corrected_df, corrected_eval_path)

    log.info("修正後 CER/WER 再評価中...")
    after_df = _evaluate_cer_wer(corrected_eval_path)

    log.info("修正前ベースライン読み込み: %s", args.baseline_eval)
    before_df = _load_baseline_cer(args.baseline_eval)

    merged = before_df.merge(
        after_df,
        on="sample_id",
        suffixes=("_before", "_after"),
        how="inner",
    )

    n_total = len(corrected_df)
    n_ok = len(ok_df)
    n_skipped = len(corrected_df[corrected_df["status"] == "skipped"])
    n_corrected = len(
        ok_df[ok_df["correction_changed"].astype(str).str.lower() == "true"],
    )
    n_unchanged = n_ok - n_corrected

    if not merged.empty:
        cer_before_vals = pd.to_numeric(merged["cer_before"], errors="coerce")
        cer_after_vals = pd.to_numeric(merged["cer_after"], errors="coerce")
        wer_before_vals = pd.to_numeric(merged["wer_before"], errors="coerce")
        wer_after_vals = pd.to_numeric(merged["wer_after"], errors="coerce")

        both_valid = cer_before_vals.notna() & cer_after_vals.notna()
        n_improved = int(
            (cer_after_vals[both_valid] < cer_before_vals[both_valid]).sum(),
        )
        n_degraded = int(
            (cer_after_vals[both_valid] > cer_before_vals[both_valid]).sum(),
        )
        n_equal = int((cer_after_vals[both_valid] == cer_before_vals[both_valid]).sum())

        corpus_cer_before = cer_before_vals.mean()
        corpus_cer_after = cer_after_vals.mean()
        corpus_wer_before = wer_before_vals.mean()
        corpus_wer_after = wer_after_vals.mean()
    else:
        n_improved = n_degraded = n_equal = 0
        corpus_cer_before = corpus_cer_after = None
        corpus_wer_before = corpus_wer_after = None

    judge_stats_before = {
        "llm_cer_mean": None,
        "intent_score_mean": None,
        "entity_preservation_mean": None,
    }
    judge_stats_after = {
        "llm_cer_mean": None,
        "intent_score_mean": None,
        "entity_preservation_mean": None,
    }
    spearman_corr = ""
    spearman_pval = ""

    if args.judge_csv.exists():
        log.info("Judge CSV 読み込み: %s", args.judge_csv)
        judge_df = pd.read_csv(args.judge_csv)
        judge_stats_before = _compute_judge_stats(judge_df, "asr_original")
        judge_stats_after = _compute_judge_stats(judge_df, "llm_corrected")

        judge_original = judge_df[
            (judge_df["target_type"] == "asr_original") & (judge_df["status"] == "ok")
        ][["sample_id", "llm_cer"]].copy()
        judge_original["llm_cer"] = pd.to_numeric(
            judge_original["llm_cer"],
            errors="coerce",
        )

        corr_df = before_df.merge(judge_original, on="sample_id", how="inner")
        cer_vals = pd.to_numeric(corr_df["cer"], errors="coerce")
        llm_cer_vals = corr_df["llm_cer"]
        both_ok = cer_vals.notna() & llm_cer_vals.notna()
        min_samples_for_spearman = 3
        if both_ok.sum() > min_samples_for_spearman:
            corr, pval = sp_stats.spearmanr(
                cer_vals[both_ok],
                llm_cer_vals[both_ok],
            )
            spearman_corr = round(corr, 4)
            spearman_pval = round(pval, 6)
            log.info(
                "Spearman(CER, LLM-CER): r=%.4f, p=%.6f",
                corr,
                pval,
            )

    summary_row = {
        "corrector_model": corrector_model,
        "n_total": n_total,
        "n_ok": n_ok,
        "n_skipped": n_skipped,
        "n_corrected": n_corrected,
        "n_unchanged": n_unchanged,
        "n_improved_cer": n_improved,
        "n_degraded_cer": n_degraded,
        "n_equal_cer": n_equal,
        "cer_before": round(corpus_cer_before, 6)
        if corpus_cer_before is not None
        else "",
        "cer_after": round(corpus_cer_after, 6) if corpus_cer_after is not None else "",
        "cer_relative_improvement_pct": _relative_improvement(
            corpus_cer_before,
            corpus_cer_after,
        ),
        "wer_before": round(corpus_wer_before, 6)
        if corpus_wer_before is not None
        else "",
        "wer_after": round(corpus_wer_after, 6) if corpus_wer_after is not None else "",
        "wer_relative_improvement_pct": _relative_improvement(
            corpus_wer_before,
            corpus_wer_after,
        ),
        "llm_cer_before": judge_stats_before["llm_cer_mean"] or "",
        "llm_cer_after": judge_stats_after["llm_cer_mean"] or "",
        "llm_cer_relative_improvement_pct": _relative_improvement(
            judge_stats_before["llm_cer_mean"],
            judge_stats_after["llm_cer_mean"],
        ),
        "intent_score_mean_before": judge_stats_before["intent_score_mean"] or "",
        "intent_score_mean_after": judge_stats_after["intent_score_mean"] or "",
        "entity_preservation_mean_before": judge_stats_before[
            "entity_preservation_mean"
        ]
        or "",
        "entity_preservation_mean_after": judge_stats_after["entity_preservation_mean"]
        or "",
        "spearman_cer_vs_llm_cer": spearman_corr,
        "spearman_pvalue": spearman_pval,
    }

    summary_path = output_dir / "phase3_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PHASE3_SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerow(summary_row)
    log.info("出力: %s", summary_path)

    config = {
        "corrected_csv": str(args.corrected_csv),
        "judge_csv": str(args.judge_csv),
        "baseline_eval": str(args.baseline_eval),
        "metadata": str(args.metadata),
        "normalization_profile": "ja_surface_v1",
        "tokenizer_split_mode": "C",
        "versions": {
            "jiwer": importlib.metadata.version("jiwer"),
            "sudachipy": importlib.metadata.version("sudachipy"),
            "pandas": importlib.metadata.version("pandas"),
            "scipy": importlib.metadata.version("scipy"),
        },
        "executed_at": datetime.now(timezone.utc).isoformat(),
    }
    config_path = output_dir / "phase3_config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    log.info("出力: %s", config_path)

    log.info("=== Phase3 Summary ===")
    log.info("Model: %s", corrector_model)
    log.info("Total: %d, OK: %d, Skipped: %d", n_total, n_ok, n_skipped)
    log.info("Corrected: %d, Unchanged: %d", n_corrected, n_unchanged)
    log.info(
        "CER:  before=%.4f, after=%.4f, improved=%d, degraded=%d, equal=%d",
        corpus_cer_before or 0,
        corpus_cer_after or 0,
        n_improved,
        n_degraded,
        n_equal,
    )
    if judge_stats_before["llm_cer_mean"] is not None:
        log.info(
            "LLM-CER: before=%.4f, after=%.4f",
            judge_stats_before["llm_cer_mean"],
            judge_stats_after["llm_cer_mean"] or 0,
        )
        log.info(
            "Intent:  before=%.4f, after=%.4f",
            judge_stats_before["intent_score_mean"] or 0,
            judge_stats_after["intent_score_mean"] or 0,
        )
        log.info(
            "Entity:  before=%.4f, after=%.4f",
            judge_stats_before["entity_preservation_mean"] or 0,
            judge_stats_after["entity_preservation_mean"] or 0,
        )


if __name__ == "__main__":
    main()
