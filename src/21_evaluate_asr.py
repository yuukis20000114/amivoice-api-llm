"""ASR 結果を CER/WER で評価する."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import jiwer
import pandas as pd

from text_normalization import (
    PROFILES,
    normalize_ja_text,
    tokenize_ja_words,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

UTTERANCE_FIELDS = [
    "sample_id",
    "audio_variant",
    "asr_provider",
    "asr_model",
    "reference_text_raw",
    "asr_text_raw",
    "reference_text_norm",
    "asr_text_norm",
    "reference_tokens",
    "asr_tokens",
    "cer_s",
    "cer_i",
    "cer_d",
    "cer_n",
    "cer",
    "wer_s",
    "wer_i",
    "wer_d",
    "wer_n",
    "wer",
    "status",
    "error_message",
    "raw_response_path",
]

SUMMARY_FIELDS = [
    "asr_provider",
    "asr_model",
    "audio_variant",
    "normalization_profile",
    "n_total",
    "n_ok",
    "n_error",
    "n_excluded_empty_ref",
    "coverage",
    "cer_errors",
    "cer_n",
    "cer",
    "wer_errors",
    "wer_n",
    "wer",
    "processing_time_sec_mean",
]

AUDIT_FIELDS = [
    "sample_id",
    "audio_variant",
    "field",
    "before",
    "after",
    "changed_reason",
]


def _load_metadata(path: Path) -> dict[str, str]:
    df = pd.read_csv(path)
    return dict(zip(df["sample_id"], df["reference_text"], strict=True))


def _load_asr_csvs(paths: list[Path]) -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in paths]
    return pd.concat(frames, ignore_index=True)


def _validate(df: pd.DataFrame, metadata: dict[str, str]) -> pd.DataFrame:
    dup_cols = ["sample_id", "audio_variant", "asr_provider", "asr_model"]
    dups = df.duplicated(subset=dup_cols, keep=False)
    if dups.any():
        n = dups.sum()
        log.error("重複キーが %d 行あります。最初の重複を表示:", n)
        log.error("\n%s", df[dups].head(10).to_string())
        sys.exit(1)

    mismatch_mask = []
    for _, row in df.iterrows():
        sid = row["sample_id"]
        meta_ref = metadata.get(sid)
        if meta_ref is None:
            log.warning("metadata に存在しない sample_id: %s", sid)
            mismatch_mask.append(True)
        elif meta_ref != row["reference_text"]:
            log.warning(
                "reference_text 不一致 sample_id=%s: metadata=%r asr=%r",
                sid,
                meta_ref,
                row["reference_text"],
            )
            mismatch_mask.append(True)
        else:
            mismatch_mask.append(False)

    n_mismatch = sum(mismatch_mask)
    if n_mismatch > 0:
        log.warning("reference_text 不一致/未検出: %d 行を除外", n_mismatch)
        df = df[~pd.Series(mismatch_mask)].reset_index(drop=True)

    return df


def _compute_sid_metrics(
    alignment_chunks: list,
) -> tuple[int, int, int, int]:
    s = sum(
        chunk.ref_end_idx - chunk.ref_start_idx
        for chunk in alignment_chunks
        if chunk.type == "substitute"
    )
    i = sum(
        chunk.hyp_end_idx - chunk.hyp_start_idx
        for chunk in alignment_chunks
        if chunk.type == "insert"
    )
    d = sum(
        chunk.ref_end_idx - chunk.ref_start_idx
        for chunk in alignment_chunks
        if chunk.type == "delete"
    )
    hits = sum(
        chunk.ref_end_idx - chunk.ref_start_idx
        for chunk in alignment_chunks
        if chunk.type == "equal"
    )
    n = hits + s + d
    return s, i, d, n


def _error_row_to_utterance(row: pd.Series) -> dict:
    empty_metrics = dict.fromkeys(
        (
            "cer_s",
            "cer_i",
            "cer_d",
            "cer_n",
            "cer",
            "wer_s",
            "wer_i",
            "wer_d",
            "wer_n",
            "wer",
        ),
        "",
    )
    return {
        "sample_id": row["sample_id"],
        "audio_variant": row["audio_variant"],
        "asr_provider": row["asr_provider"],
        "asr_model": row["asr_model"],
        "reference_text_raw": row["reference_text"],
        "asr_text_raw": row.get("asr_text", ""),
        "reference_text_norm": "",
        "asr_text_norm": "",
        "reference_tokens": "",
        "asr_tokens": "",
        **empty_metrics,
        "status": row["status"],
        "error_message": row.get("error_message", ""),
        "raw_response_path": row.get("raw_response_path", ""),
    }


def _collect_audit_rows(  # noqa: PLR0913
    ok_df: pd.DataFrame,
    refs_raw: list[str],
    refs_norm: list[str],
    hyps_raw: list[str],
    hyps_norm: list[str],
    profile: str,
) -> list[dict]:
    audit_rows: list[dict] = []
    for idx in range(len(refs_raw)):
        sid = ok_df.iloc[idx]["sample_id"]
        variant = ok_df.iloc[idx]["audio_variant"]
        if refs_raw[idx] != refs_norm[idx]:
            audit_rows.append(
                {
                    "sample_id": sid,
                    "audio_variant": variant,
                    "field": "reference_text",
                    "before": refs_raw[idx],
                    "after": refs_norm[idx],
                    "changed_reason": profile,
                },
            )
        if hyps_raw[idx] != hyps_norm[idx]:
            audit_rows.append(
                {
                    "sample_id": sid,
                    "audio_variant": variant,
                    "field": "asr_text",
                    "before": hyps_raw[idx],
                    "after": hyps_norm[idx],
                    "changed_reason": profile,
                },
            )
    return audit_rows


def _fill_metrics(rec: dict, cer_align: list, wer_align: list) -> None:
    cer_s, cer_i, cer_d, cer_n = _compute_sid_metrics(cer_align)
    wer_s, wer_i, wer_d, wer_n = _compute_sid_metrics(wer_align)
    rec["cer_s"] = cer_s
    rec["cer_i"] = cer_i
    rec["cer_d"] = cer_d
    rec["cer_n"] = cer_n
    rec["cer"] = (cer_s + cer_i + cer_d) / cer_n if cer_n > 0 else 0.0
    rec["wer_s"] = wer_s
    rec["wer_i"] = wer_i
    rec["wer_d"] = wer_d
    rec["wer_n"] = wer_n
    rec["wer"] = (wer_s + wer_i + wer_d) / wer_n if wer_n > 0 else 0.0


def _evaluate_utterances(
    df: pd.DataFrame,
    profile: str,
    split_mode: str,
) -> tuple[list[dict], list[dict]]:
    ok_df = df[df["status"] == "ok"].copy()
    error_df = df[df["status"] != "ok"]

    utterances = [_error_row_to_utterance(row) for _, row in error_df.iterrows()]

    if ok_df.empty:
        return utterances, []

    refs_raw = ok_df["reference_text"].fillna("").tolist()
    hyps_raw = ok_df["asr_text"].fillna("").tolist()
    refs_norm = [normalize_ja_text(t, profile) for t in refs_raw]
    hyps_norm = [normalize_ja_text(t, profile) for t in hyps_raw]

    audit_rows = _collect_audit_rows(
        ok_df,
        refs_raw,
        refs_norm,
        hyps_raw,
        hyps_norm,
        profile,
    )

    refs_tokens = [tokenize_ja_words(t, split_mode) for t in refs_norm]
    hyps_tokens = [tokenize_ja_words(t, split_mode) for t in hyps_norm]
    refs_words_str = [" ".join(toks) for toks in refs_tokens]
    hyps_words_str = [" ".join(toks) for toks in hyps_tokens]

    eval_indices = [i for i, r in enumerate(refs_norm) if len(r) > 0]
    skip_indices = set(range(len(refs_norm))) - set(eval_indices)

    cer_results = None
    wer_results = None
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
        rec: dict = {
            "sample_id": row["sample_id"],
            "audio_variant": row["audio_variant"],
            "asr_provider": row["asr_provider"],
            "asr_model": row["asr_model"],
            "reference_text_raw": refs_raw[idx],
            "asr_text_raw": hyps_raw[idx],
            "reference_text_norm": refs_norm[idx],
            "asr_text_norm": hyps_norm[idx],
            "reference_tokens": " ".join(refs_tokens[idx]),
            "asr_tokens": " ".join(hyps_tokens[idx]),
            "status": "ok",
            "error_message": "",
            "raw_response_path": row.get("raw_response_path", ""),
        }
        if idx in skip_indices:
            rec["status"] = "excluded_empty_ref"
            for k in (
                "cer_s",
                "cer_i",
                "cer_d",
                "cer_n",
                "cer",
                "wer_s",
                "wer_i",
                "wer_d",
                "wer_n",
                "wer",
            ):
                rec[k] = ""
        else:
            _fill_metrics(
                rec,
                cer_results.alignments[eval_pos],
                wer_results.alignments[eval_pos],
            )
            eval_pos += 1
        utterances.append(rec)

    return utterances, audit_rows


def _build_summary(
    utterances: list[dict],
    profile: str,
    asr_df: pd.DataFrame,
) -> list[dict]:
    processing_times: dict[tuple, list[float]] = {}
    for _, row in asr_df.iterrows():
        key = (row["asr_provider"], row["asr_model"], row["audio_variant"])
        t = row.get("processing_time_sec")
        if pd.notna(t) and t != "":
            processing_times.setdefault(key, []).append(float(t))

    groups: dict[tuple, list[dict]] = {}
    for utt in utterances:
        key = (utt["asr_provider"], utt["asr_model"], utt["audio_variant"])
        groups.setdefault(key, []).append(utt)

    summaries = []
    for (provider, model, variant), utts in sorted(groups.items()):
        n_total = len(utts)
        n_ok = sum(1 for u in utts if u["status"] == "ok")
        n_error = sum(
            1 for u in utts if u["status"] not in ("ok", "excluded_empty_ref")
        )
        n_excluded_empty_ref = sum(
            1 for u in utts if u["status"] == "excluded_empty_ref"
        )

        eval_utts = [u for u in utts if u["status"] == "ok"]
        cer_s_sum = sum(u["cer_s"] for u in eval_utts)
        cer_i_sum = sum(u["cer_i"] for u in eval_utts)
        cer_d_sum = sum(u["cer_d"] for u in eval_utts)
        cer_n_sum = sum(u["cer_n"] for u in eval_utts)
        wer_s_sum = sum(u["wer_s"] for u in eval_utts)
        wer_i_sum = sum(u["wer_i"] for u in eval_utts)
        wer_d_sum = sum(u["wer_d"] for u in eval_utts)
        wer_n_sum = sum(u["wer_n"] for u in eval_utts)

        cer_errors = cer_s_sum + cer_i_sum + cer_d_sum
        wer_errors = wer_s_sum + wer_i_sum + wer_d_sum
        corpus_cer = cer_errors / cer_n_sum if cer_n_sum > 0 else 0.0
        corpus_wer = wer_errors / wer_n_sum if wer_n_sum > 0 else 0.0

        times = processing_times.get((provider, model, variant), [])
        pt_mean = sum(times) / len(times) if times else 0.0

        summaries.append(
            {
                "asr_provider": provider,
                "asr_model": model,
                "audio_variant": variant,
                "normalization_profile": profile,
                "n_total": n_total,
                "n_ok": n_ok,
                "n_error": n_error,
                "n_excluded_empty_ref": n_excluded_empty_ref,
                "coverage": n_ok / n_total if n_total > 0 else 0.0,
                "cer_errors": cer_errors,
                "cer_n": cer_n_sum,
                "cer": corpus_cer,
                "wer_errors": wer_errors,
                "wer_n": wer_n_sum,
                "wer": corpus_wer,
                "processing_time_sec_mean": round(pt_mean, 4),
            },
        )

    return summaries


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    log.info("出力: %s (%d 行)", path, len(rows))


def _write_config(
    path: Path,
    args: argparse.Namespace,
    profile: str,
    split_mode: str,
) -> None:
    import importlib.metadata  # noqa: PLC0415

    config = {
        "asr_csv_paths": [str(p) for p in args.asr_csv],
        "metadata_path": str(args.metadata),
        "normalization_profile": profile,
        "tokenizer": "sudachipy",
        "tokenizer_split_mode": split_mode,
        "versions": {
            "jiwer": importlib.metadata.version("jiwer"),
            "sudachipy": importlib.metadata.version("sudachipy"),
            "sudachidict_core": importlib.metadata.version("sudachidict-core"),
        },
        "executed_at": datetime.now(timezone.utc).isoformat(),
        "command_args": sys.argv,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    log.info("出力: %s", path)


def main() -> None:
    parser = argparse.ArgumentParser(description="ASR 結果を CER/WER で評価する")
    parser.add_argument(
        "--asr-csv",
        type=Path,
        nargs="+",
        required=True,
        help="ASR 結果 CSV パス (複数可)",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        required=True,
        help="metadata CSV パス",
    )
    parser.add_argument(
        "--normalization-profile",
        default="ja_surface_v1",
        choices=sorted(PROFILES),
        help="正規化プロファイル (default: ja_surface_v1)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/evaluations"),
        help="出力ディレクトリ (default: outputs/evaluations)",
    )
    parser.add_argument(
        "--include-overall",
        action="store_true",
        help="全 variant 混合の総合行を追加",
    )
    args = parser.parse_args()

    profile = args.normalization_profile
    split_mode = "C"
    output_dir = args.output_dir

    log.info("metadata 読み込み: %s", args.metadata)
    metadata = _load_metadata(args.metadata)
    log.info("metadata: %d 件", len(metadata))

    log.info("ASR CSV 読み込み: %s", [str(p) for p in args.asr_csv])
    asr_df = _load_asr_csvs(args.asr_csv)
    log.info("ASR CSV: %d 行", len(asr_df))

    asr_df = _validate(asr_df, metadata)

    log.info("評価開始 (profile=%s, split_mode=%s)", profile, split_mode)
    utterances, audit_rows = _evaluate_utterances(asr_df, profile, split_mode)

    summaries = _build_summary(utterances, profile, asr_df)

    _write_csv(
        output_dir / "asr_eval_utterances.csv",
        utterances,
        UTTERANCE_FIELDS,
    )
    _write_csv(
        output_dir / "asr_eval_summary.csv",
        summaries,
        SUMMARY_FIELDS,
    )
    _write_csv(
        output_dir / "asr_eval_normalization_audit.csv",
        audit_rows,
        AUDIT_FIELDS,
    )
    _write_config(
        output_dir / "asr_eval_config.json",
        args,
        profile,
        split_mode,
    )

    log.info("=== Summary ===")
    for s in summaries:
        log.info(
            "%s/%s [%s] CER=%.4f WER=%.4f (n_ok=%d, n_error=%d, n_excluded=%d)",
            s["asr_provider"],
            s["asr_model"],
            s["audio_variant"],
            s["cer"],
            s["wer"],
            s["n_ok"],
            s["n_error"],
            s["n_excluded_empty_ref"],
        )


if __name__ == "__main__":
    main()
