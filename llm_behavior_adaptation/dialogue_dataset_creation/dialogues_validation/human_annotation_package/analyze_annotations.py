#!/usr/bin/env python3
"""
Analyze human annotation CSVs and compare with LLM judge outputs.

Expected input:
- One or more completed CSV files (same schema as *_to_fill.csv), concatenatable.
Optional:
- A column named LLM_Judge (numeric) or LLM_Judge_<metric> to compare correlations.

Outputs:
- Per-metric mean/std
- Inter-annotator agreement (ICC(2,k) approximation via pingouin if available; fallback to pairwise correlations)
- Human vs LLM correlations (Pearson/Spearman)
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
from scipy import stats

METRICS = ["Coverage_0to5", "Correctness_0to5", "Diversity_1to5", "Relevance_1to5", "Naturalness_1to5"]


def load(paths):
    dfs = []
    for p in paths:
        df = pd.read_csv(p)
        df["__source__"] = os.path.basename(p)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def basic_stats(df):
    out = {}
    for m in METRICS:
        s = pd.to_numeric(df[m], errors="coerce")
        out[m] = {"n": int(s.notna().sum()), "mean": float(s.mean()), "std": float(s.std(ddof=1))}
    return out


def check_constraints(df):
    bad = df[
        pd.to_numeric(df["Correctness_0to5"], errors="coerce") > pd.to_numeric(df["Coverage_0to5"], errors="coerce")
    ]
    return bad[["Sample_ID", "Annotator_ID", "Coverage_0to5", "Correctness_0to5", "__source__"]]


def pivot_for_icc(df, metric):
    # long -> wide by Sample_ID x Annotator_ID
    tmp = df[["Sample_ID", "Annotator_ID", metric]].copy()
    tmp[metric] = pd.to_numeric(tmp[metric], errors="coerce")
    wide = tmp.pivot_table(index="Sample_ID", columns="Annotator_ID", values=metric, aggfunc="mean")
    return wide


def icc_fallback(wide):
    # simple average pairwise Pearson across annotators, ignoring NaNs
    cols = list(wide.columns)
    pairs = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a = wide[cols[i]]
            b = wide[cols[j]]
            mask = a.notna() & b.notna()
            if mask.sum() >= 5:
                r = np.corrcoef(a[mask], b[mask])[0, 1]
                pairs.append(r)
    return float(np.nanmean(pairs)) if pairs else float("nan")


def human_vs_llm(df):
    llm_cols = [c for c in df.columns if c.startswith("LLM_Judge")]
    results = {}
    if not llm_cols:
        return results
    # aggregate human per Sample_ID
    for m in METRICS:
        human = df.groupby("Sample_ID")[m].mean(numeric_only=True)
        for c in llm_cols:
            llm = df.groupby("Sample_ID")[c].mean(numeric_only=True)
            joined = pd.concat([human, llm], axis=1, join="inner").dropna()
            if joined.shape[0] < 10:
                continue
            pear = stats.pearsonr(joined.iloc[:, 0], joined.iloc[:, 1])
            spear = stats.spearmanr(joined.iloc[:, 0], joined.iloc[:, 1])
            results[f"{m} vs {c}"] = {
                "n": int(joined.shape[0]),
                "pearson_r": float(pear.statistic),
                "pearson_p": float(pear.pvalue),
                "spearman_r": float(spear.statistic),
                "spearman_p": float(spear.pvalue),
            }
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--glob",
        required=True,
        help="Glob for completed CSVs, e.g. 'investment_50/*_to_fill_completed.csv' or 'investment_50/*to_fill.csv'",
    )
    ap.add_argument("--out", default="analysis_report.json")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.glob))
    if not paths:
        raise SystemExit(f"No files matched: {args.glob}")

    df = load(paths)

    report = {}
    report["files"] = paths
    report["basic_stats"] = basic_stats(df)

    bad = check_constraints(df)
    report["violations_correctness_gt_coverage"] = bad.to_dict(orient="records")

    report["agreement_pairwise_mean_r"] = {}
    for m in METRICS:
        wide = pivot_for_icc(df, m)
        report["agreement_pairwise_mean_r"][m] = icc_fallback(wide)

    report["human_vs_llm"] = human_vs_llm(df)

    with open(args.out, "w") as f:
        import json

        json.dump(report, f, indent=2)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
