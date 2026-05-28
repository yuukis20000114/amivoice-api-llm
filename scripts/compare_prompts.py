"""v1 / v2 / v2r 訂正結果の比較分析スクリプト."""

from __future__ import annotations

import csv
import json
import os
import sys
from difflib import SequenceMatcher


def load(path: str) -> dict[str, dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return {r["sample_id"]: r for r in csv.DictReader(f)}


def sim(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def token_stats(raw_dir: str, sample_ids: list[str]) -> tuple[int, int]:
    inp = out = 0
    for sid in sample_ids:
        p = os.path.join(raw_dir, f"{sid}.json")
        if os.path.exists(p):
            with open(p) as f:
                d = json.load(f)
            inp += d.get("input_tokens", 0)
            out += d.get("output_tokens", 0)
    return inp, out


def main() -> None:
    dirs = {
        "v1": "outputs/llm_correction_v1_100",
        "v2": "outputs/llm_correction_v2_100",
        "v2r": "outputs/llm_correction_v2r_100",
    }
    datasets = {}
    for label, d in dirs.items():
        csv_path = os.path.join(d, "corrected_amivoice.csv")
        if not os.path.exists(csv_path):
            continue
        datasets[label] = load(csv_path)

    if len(datasets) < 2:
        print("Need at least 2 datasets to compare")
        sys.exit(1)

    base = datasets[list(datasets.keys())[0]]
    sids = list(base.keys())

    print("=" * 80)
    print("総合比較")
    print("=" * 80)

    for label, data in datasets.items():
        changed = sum(1 for r in data.values() if r["correction_changed"] == "true")
        avg_sim = sum(
            sim(r["reference_text"], r["corrected_text"]) for r in data.values()
        ) / len(data)
        avg_asr_sim = sum(
            sim(r["reference_text"], r["asr_text_original"]) for r in data.values()
        ) / len(data)
        raw_dir = os.path.join(dirs[label], "correction_raw")
        inp, out = token_stats(raw_dir, sids)
        print(
            f"  {label:>4}: changed={changed:>3}/100  "
            f"avg_sim={avg_sim:.4f} (ASR={avg_asr_sim:.4f})  "
            f"tokens={inp+out:>6,}"
        )

    # Pairwise comparison for all combinations
    labels = list(datasets.keys())
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            la, lb = labels[i], labels[j]
            da, db = datasets[la], datasets[lb]
            better_a = better_b = equal = 0
            for sid in sids:
                ref = da[sid]["reference_text"]
                sa = sim(ref, da[sid]["corrected_text"])
                sb = sim(ref, db[sid]["corrected_text"])
                if sb > sa + 0.005:
                    better_b += 1
                elif sa > sb + 0.005:
                    better_a += 1
                else:
                    equal += 1
            print(f"\n  {la} vs {lb}: {la}勝={better_a}, {lb}勝={better_b}, 同等={equal}")

    # Detailed: cases where v2r differs from both v1 and v2
    if "v2r" in datasets and "v1" in datasets:
        d1 = datasets["v1"]
        d2r = datasets["v2r"]
        print(f"\n{'='*80}")
        print("v2r vs v1 詳細（差異のあるケースのみ）")
        print(f"{'='*80}")

        diffs = []
        for sid in sids:
            ref = d1[sid]["reference_text"]
            asr = d1[sid]["asr_text_original"]
            c1 = d1[sid]["corrected_text"]
            c2r = d2r[sid]["corrected_text"]
            if c1 == c2r:
                continue
            s1 = sim(ref, c1)
            s2r = sim(ref, c2r)
            diffs.append((sid, ref, asr, c1, c2r, s1, s2r))

        diffs.sort(key=lambda x: x[6] - x[5], reverse=True)
        for sid, ref, asr, c1, c2r, s1, s2r in diffs:
            delta = s2r - s1
            mark = "✓" if delta > 0.005 else ("✗" if delta < -0.005 else "→")
            print(
                f"\n{mark} [{sid}] v1→REF={s1:.3f} v2r→REF={s2r:.3f} ({delta:+.3f})"
            )
            print(f"  REF: {ref}")
            print(f"  ASR: {asr}")
            print(f"  v1 : {c1}")
            print(f"  v2r: {c2r}")

    # Check previous v2 regressions
    if "v2" in datasets and "v2r" in datasets and "v1" in datasets:
        print(f"\n{'='*80}")
        print("前回v2退行13件の v2r での結果")
        print(f"{'='*80}")
        d1 = datasets["v1"]
        d2 = datasets["v2"]
        d2r = datasets["v2r"]
        for sid in sids:
            ref = d1[sid]["reference_text"]
            s1 = sim(ref, d1[sid]["corrected_text"])
            s2 = sim(ref, d2[sid]["corrected_text"])
            s2r = sim(ref, d2r[sid]["corrected_text"])
            if s1 > s2 + 0.005:
                fixed = "✓修正" if s2r >= s1 - 0.005 else "✗残存"
                print(
                    f"  {fixed} [{sid}] v1={s1:.3f} v2={s2:.3f} v2r={s2r:.3f}"
                    f"  REF: {ref[:40]}..."
                )


if __name__ == "__main__":
    main()
