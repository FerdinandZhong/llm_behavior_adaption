#!/usr/bin/env python3
"""
IAA v2 – with Prolific demographics, annotator filtering, and human vs. LLM analysis.

Changes from v1:
  - Match Qualtrics respondents to Prolific demographics via start timestamp (±30 s)
  - Career exclusions: Ann4 (too strict, Coverage mean ≈1.9) + Ann6 (straight-liner, std=0)
  - Ann3 career flagged CONSENT_REVOKED from Prolific (user notified)
  - Recompute IAA on 4 valid annotators per topic
  - Detailed human-mean vs. LLM-judge per-metric comparison
"""

import argparse
import json
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=stats.ConstantInputWarning)
warnings.filterwarnings("ignore", "invalid value encountered")
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── constants ────────────────────────────────────────────────────────────────
METRICS = ["Coverage", "Correctness", "Diversity", "Relevance", "Naturalness"]
Q_OFFSET = {m: i for i, m in enumerate(METRICS)}  # 0–4
Q_BASE = 14
Q_STRIDE = 8

# Career A actual block order (extracted from Career-A QSF — Dialogue 01 … 25)
ANN1_SAMPLES = [
    496070280,
    704070203,
    68070938,
    124071090,
    410071020,
    364071290,
    702070435,
    608070387,
    344071848,
    344070673,
    276070154,
    368070918,
    764070961,
    716070584,
    170074611,
    792070837,
    762070386,
    51070749,
    792071478,
    840071786,
    158070083,
    558070712,
    124072218,
    344071377,
    792071442,
]

# Career Step 2 — 25 samples from Career-B survey (QSF block order = CSV row order)
ANN2_CAREER_SAMPLES = [
    642071000,
    392071228,
    360071238,
    68070117,
    68071656,
    604070788,
    50070775,
    566070301,
    50070388,
    704070277,
    484070289,
    586071298,
    762071190,
    50070454,
    300071162,
    364071175,
    702071860,
    124073038,
    586071419,
    364072061,
    360070714,
    203070285,
    586070646,
    792072031,
    50070574,
]

# Investment Step 2 — 25 samples from Investment-B survey (QSF block order = CSV row order)
ANN2_INVEST_SAMPLES = [
    642071000,
    392071228,
    360071238,
    68070117,
    68071656,
    604070788,
    50070775,
    566070301,
    50070388,
    704070277,
    484070289,
    586071298,
    762071190,
    50070454,
    300071162,
    364071175,
    702071860,
    124073038,
    586071419,
    364072061,
    360070714,
    203070285,
    586070646,
    792072031,
    50070574,
]

LLM_KEY = {
    "Coverage": "coverage",
    "Correctness": "correctness",
    "Diversity": "diversity",
    "Relevance": "relevance",
    "Naturalness": "naturalness",
}

BASE = Path(__file__).parent


# ── Prolific helpers ──────────────────────────────────────────────────────────


def load_prolific(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["start_utc"] = pd.to_datetime(df["Started at"], utc=True)
    return df


def match_prolific(qualtrics_df: pd.DataFrame, prolific_df: pd.DataFrame, tol_sec: int = 30) -> pd.DataFrame:
    """
    Match each Qualtrics real respondent to a Prolific row by nearest start time.
    Adds Prolific demographic columns.  Uses HKT→UTC conversion for Qualtrics.
    """
    q = qualtrics_df.copy()
    q["start_utc"] = pd.to_datetime(q["StartDate"]).dt.tz_localize("Asia/Hong_Kong").dt.tz_convert("UTC")

    demo_cols = [
        "Participant id",
        "Status",
        "Age",
        "Sex",
        "Country of residence",
        "Nationality",
        "Employment status",
        "Student status",
    ]

    matched_rows = []
    for _, qrow in q.iterrows():
        diffs = (prolific_df["start_utc"] - qrow["start_utc"]).abs()
        best_idx = diffs.idxmin()
        if diffs[best_idx].total_seconds() <= tol_sec:
            prow = prolific_df.loc[best_idx, demo_cols].to_dict()
        else:
            prow = {c: "UNMATCHED" for c in demo_cols}
        matched_rows.append(prow)

    demo_df = pd.DataFrame(matched_rows, index=q.index)
    return pd.concat([q.reset_index(drop=True), demo_df.reset_index(drop=True)], axis=1)


# ── parsing helpers ───────────────────────────────────────────────────────────


def parse_qualtrics_tidy(
    csv_path: str,
    sample_ids: list,
    exclude_rows: list = None,
    step2_csv_path: str = None,
    step2_sample_ids: list = None,
    step2_exclude_rows: list = None,
) -> pd.DataFrame:
    """Return tidy df, optionally merging a Step 2 export for 50-sample coverage.

    Both Step 1 and Step 2 surveys use the same Q-numbering scheme
    (Q_BASE=14, Q_STRIDE=8), so the same parsing logic applies to both.
    step2_exclude_rows allows separate exclusion criteria for the Step 2 batch
    (e.g. to drop a duplicate submission without excluding valid annotators).
    """

    def _parse_one(path, s_ids, exclude):
        raw = pd.read_csv(path)
        real = raw[raw["Status"] == "IP Address"].reset_index(drop=True)
        recs = []
        for resp_idx, row in real.iterrows():
            if exclude and resp_idx in exclude:
                continue
            ann_label = f"Ann{resp_idx + 1}"
            for block_idx, sample_id in enumerate(s_ids):
                for metric, offset in Q_OFFSET.items():
                    q_num = Q_BASE + Q_STRIDE * block_idx + offset
                    val = pd.to_numeric(row.get(f"Q{q_num}"), errors="coerce")
                    recs.append(
                        {
                            "respondent": ann_label,
                            "resp_idx": resp_idx,
                            "sample_id": sample_id,
                            "metric": metric,
                            "score": val,
                        }
                    )
        return recs

    records = _parse_one(csv_path, sample_ids, exclude_rows)

    if step2_csv_path and Path(step2_csv_path).exists() and step2_sample_ids:
        s2_exclude = step2_exclude_rows if step2_exclude_rows is not None else exclude_rows
        records += _parse_one(step2_csv_path, step2_sample_ids, s2_exclude)

    return pd.DataFrame(records)


def load_llm(jsonl_path: str, sample_ids: set) -> pd.DataFrame:
    records = []
    with open(jsonl_path) as f:
        for line in f:
            obj = json.loads(line)
            sid = int(obj["dialogue_index"])
            if sid not in sample_ids:
                continue
            avg = obj.get("avg_scores", obj.get("scores", {}))
            for m in METRICS:
                records.append({"sample_id": sid, "metric": m, "llm_score": float(avg.get(LLM_KEY[m], np.nan))})
    return pd.DataFrame(records)


# ── IAA helpers ───────────────────────────────────────────────────────────────


def wide(tidy: pd.DataFrame, metric: str) -> pd.DataFrame:
    return tidy[tidy["metric"] == metric].pivot_table(
        index="sample_id", columns="respondent", values="score", aggfunc="mean"
    )


def icc(wide_df: pd.DataFrame) -> dict:
    try:
        import pingouin as pg

        long = wide_df.reset_index().melt(id_vars="sample_id", var_name="rater", value_name="rating")
        long = long.dropna(subset=["rating"])
        icc_df = pg.intraclass_corr(
            data=long, targets="sample_id", raters="rater", ratings="rating", nan_policy="omit"
        )
        return {
            "icc21": round(float(icc_df.loc[icc_df["Type"] == "ICC2", "ICC"].values[0]), 4),
            "icc2k": round(float(icc_df.loc[icc_df["Type"] == "ICC2k", "ICC"].values[0]), 4),
        }
    except Exception:
        cols = list(wide_df.columns)
        rs = [
            np.corrcoef(
                wide_df.loc[wide_df[a].notna() & wide_df[b].notna(), a],
                wide_df.loc[wide_df[a].notna() & wide_df[b].notna(), b],
            )[0, 1]
            for a, b in combinations(cols, 2)
            if (wide_df[a].notna() & wide_df[b].notna()).sum() >= 5
        ]
        return {"mean_r": round(float(np.nanmean(rs)), 4) if rs else np.nan}


def kripp(wide_df: pd.DataFrame) -> float | None:
    try:
        import krippendorff

        return round(float(krippendorff.alpha(reliability_data=wide_df.values.T, level_of_measurement="ordinal")), 4)
    except Exception:
        return None


def pairwise(wide_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for a, b in combinations(list(wide_df.columns), 2):
        mask = wide_df[a].notna() & wide_df[b].notna()
        n = mask.sum()
        if n < 5:
            continue
        x, y = wide_df.loc[mask, a].values, wide_df.loc[mask, b].values
        if np.std(x) == 0 or np.std(y) == 0:
            rows.append({"ann_a": a, "ann_b": b, "n": n, "pearson_r": np.nan, "spearman_r": np.nan})
            continue
        pr, _ = stats.pearsonr(x, y)
        sr, _ = stats.spearmanr(x, y)
        rows.append({"ann_a": a, "ann_b": b, "n": n, "pearson_r": round(pr, 4), "spearman_r": round(sr, 4)})
    return pd.DataFrame(rows)


# ── human vs LLM ─────────────────────────────────────────────────────────────


def human_vs_llm(tidy: pd.DataFrame, llm_df: pd.DataFrame) -> pd.DataFrame:
    human_mean = tidy.groupby(["sample_id", "metric"])["score"].mean().reset_index(name="human_mean")
    merged = human_mean.merge(llm_df, on=["sample_id", "metric"], how="inner")

    rows = []
    for metric, grp in merged.groupby("metric"):
        valid = grp.dropna(subset=["human_mean", "llm_score"])
        n = len(valid)
        if n < 5:
            continue
        h, llm_vals = valid["human_mean"].values, valid["llm_score"].values
        llm_std = np.std(llm_vals)
        # correlation only meaningful if LLM has variance
        if llm_std < 0.01:
            pr, pp, sr, sp = np.nan, np.nan, np.nan, np.nan
        else:
            pr, pp = stats.pearsonr(h, llm_vals)
            sr, sp = stats.spearmanr(h, llm_vals)
        rows.append(
            {
                "metric": metric,
                "n": n,
                "human_mean": round(float(h.mean()), 3),
                "human_std": round(float(np.std(h)), 3),
                "llm_mean": round(float(llm_vals.mean()), 3),
                "llm_std": round(float(llm_std), 3),
                "bias(llm-human)": round(float((llm_vals - h).mean()), 4),
                "mae": round(float(np.abs(llm_vals - h).mean()), 4),
                "pearson_r": round(float(pr), 4) if not np.isnan(pr) else "N/A (const.)",
                "pearson_p": round(float(pp), 4) if not np.isnan(pp) else "",
                "spearman_r": round(float(sr), 4) if not np.isnan(sr) else "N/A (const.)",
            }
        )
    return pd.DataFrame(rows)


def per_sample_comparison(tidy: pd.DataFrame, llm_df: pd.DataFrame) -> pd.DataFrame:
    """Per-sample human-mean vs LLM score, useful for scatter/diagnosis."""
    human_mean = tidy.groupby(["sample_id", "metric"])["score"].mean().reset_index(name="human_mean")
    return human_mean.merge(llm_df, on=["sample_id", "metric"], how="inner")


# ── main ─────────────────────────────────────────────────────────────────────


def analyze(
    topic: str,
    csv_path: str,
    llm_jsonl: str,
    prolific_path: str,
    exclude_rows: list,
    flag_consent_rows: list,
    step2_csv_path: str = None,
    step2_sample_ids: list = None,
    step2_exclude_rows: list = None,
) -> dict:

    print(f"\n{'='*65}")
    print(
        f"  {topic.upper()} — exclude rows {exclude_rows}  |  step2 exclude {step2_exclude_rows}  |  flag consent {flag_consent_rows}"
    )
    print(f"{'='*65}")

    # ── match Prolific demographics ──────────────────────────────────────────
    raw = pd.read_csv(csv_path)
    real_raw = raw[raw["Status"] == "IP Address"].reset_index(drop=True)
    prolific = load_prolific(prolific_path)
    real_enriched = match_prolific(real_raw, prolific)

    print("\n── Annotator–Prolific demographic mapping ──")
    for i, row in real_enriched.iterrows():
        status = row.get("Status_y", row.get("Status", ""))
        flag = (
            " ← EXCLUDED (straight-liner)"
            if i in exclude_rows and i == max(exclude_rows or [999])
            else (
                " ← EXCLUDED (too strict)"
                if i in exclude_rows
                else " ← ⚠ CONSENT_REVOKED (Prolific)" if i in flag_consent_rows else ""
            )
        )
        print(
            f"  Ann{i+1}: {row.get('Age','?')}yo {row.get('Sex','?')}, "
            f"{row.get('Country of residence','?')}, {row.get('Employment status','?')}"
            f"  [Prolific: {status}]{flag}"
        )

    # ── parse ratings (excluding flagged rows) ──────────────────────────────
    tidy = parse_qualtrics_tidy(
        csv_path,
        ANN1_SAMPLES,
        exclude_rows=exclude_rows,
        step2_csv_path=step2_csv_path,
        step2_sample_ids=step2_sample_ids,
        step2_exclude_rows=step2_exclude_rows,
    )
    n_ann = tidy["respondent"].nunique()
    print(f"\n  Included annotators: {n_ann}  |  Samples: {tidy['sample_id'].nunique()}")

    # ── per-annotator stats ─────────────────────────────────────────────────
    print("\n── Per-annotator descriptive statistics ──")
    pivot = tidy.groupby(["respondent", "metric"])["score"].agg(mean="mean", std="std").reset_index()
    pivot_mean = pivot.pivot_table(index="respondent", columns="metric", values="mean")[METRICS].round(2)
    pivot_std = pivot.pivot_table(index="respondent", columns="metric", values="std")[METRICS].round(2)
    print("  Means:")
    print(pivot_mean.to_string())
    print("  Std devs:")
    print(pivot_std.to_string())

    # ── IAA per metric ──────────────────────────────────────────────────────
    print("\n── Inter-Annotator Agreement ──")
    iaa_rows = []
    for metric in METRICS:
        w = wide(tidy, metric)
        pw = pairwise(w)
        ic = icc(w)
        ka = kripp(w)
        mean_pr = pw["pearson_r"].mean() if not pw.empty else np.nan
        mean_sr = pw["spearman_r"].mean() if not pw.empty else np.nan
        icc_str = (
            f"ICC2(1)={ic.get('icc21','')}  ICC2(k)={ic.get('icc2k','')}"
            if "icc21" in ic
            else f"mean_r={ic.get('mean_r','')}"
        )
        print(f"  {metric:<12}  Pearson={mean_pr:.4f}  Spearman={mean_sr:.4f}  " f"{icc_str}  Kripp-α={ka}")
        iaa_rows.append(
            {
                "metric": metric,
                "mean_pearson": round(mean_pr, 4),
                "mean_spearman": round(mean_sr, 4),
                **ic,
                "krippendorff_alpha": ka,
            }
        )

    # full pairwise for Naturalness (most discriminative)
    w_nat = wide(tidy, "Naturalness")
    pw_nat = pairwise(w_nat)
    if not pw_nat.empty:
        print("\n  Full pairwise (Naturalness):")
        print(pw_nat[["ann_a", "ann_b", "n", "pearson_r", "spearman_r"]].to_string(index=False))

    # ── human vs LLM ────────────────────────────────────────────────────────
    result_hvl = None
    result_ps = None
    all_samples = list(set(ANN1_SAMPLES) | set(step2_sample_ids or []))
    if llm_jsonl and Path(llm_jsonl).exists():
        llm_df = load_llm(llm_jsonl, set(all_samples))
        if not llm_df.empty:
            hvl = human_vs_llm(tidy, llm_df)
            ps = per_sample_comparison(tidy, llm_df)
            result_hvl = hvl
            result_ps = ps

            print("\n── Human mean vs. LLM judge ──")
            print(
                hvl[
                    [
                        "metric",
                        "n",
                        "human_mean",
                        "human_std",
                        "llm_mean",
                        "llm_std",
                        "bias(llm-human)",
                        "mae",
                        "pearson_r",
                        "spearman_r",
                    ]
                ].to_string(index=False)
            )

            # Show top / bottom discrepancies
            print("\n  Top-5 samples where LLM overestimates human most (Naturalness):")
            nat = ps[ps["metric"] == "Naturalness"].copy()
            nat["gap"] = nat["llm_score"] - nat["human_mean"]
            print(nat.nlargest(5, "gap")[["sample_id", "human_mean", "llm_score", "gap"]].to_string(index=False))
            print("\n  Top-5 samples where LLM underestimates human (Naturalness):")
            print(nat.nsmallest(5, "gap")[["sample_id", "human_mean", "llm_score", "gap"]].to_string(index=False))

    return {
        "topic": topic,
        "n_annotators": n_ann,
        "excluded_row_indices": exclude_rows,
        "consent_revoked_row_indices": flag_consent_rows,
        "iaa": iaa_rows,
        "human_vs_llm": result_hvl.to_dict("records") if result_hvl is not None else None,
        "per_sample": result_ps.to_dict("records") if result_ps is not None else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="iaa_v2_report.json")
    args = ap.parse_args()

    repo_root = BASE.parent.parent.parent
    llm_dir = repo_root / "wvs_values_results/dialogue_validation"

    results = {}

    # ── career: Step 1 — exclude Ann3 (row 2, CONSENT_REVOKED) + Ann4 (row 3, too strict)
    #           Step 2 — exclude row 1 only (duplicate submission by same Prolific worker)
    results["career"] = analyze(
        topic="career",
        csv_path=str(BASE / "career_export/Dialogue Dataset Review - Career - A_March 14, 2026_11.25.csv"),
        llm_jsonl=str(llm_dir / "gpt4.1_batch_evaluation_career.jsonl"),
        prolific_path=str(BASE / "prolific_export_career.csv"),
        exclude_rows=[2, 3],  # Ann3 CONSENT_REVOKED + Ann4 too strict
        flag_consent_rows=[],
        step2_csv_path=str(BASE / "career_export/Dialogue Quality Evaluation - Career - B_March 17, 2026_22.37.csv"),
        step2_sample_ids=ANN2_CAREER_SAMPLES,
        step2_exclude_rows=[1],  # row 1 is a duplicate submission by same worker as row 0
    )

    # ── investment: all 4 annotators valid
    results["investment"] = analyze(
        topic="investment",
        csv_path=str(BASE / "investment_export/Dialogue Dataset Review - Investment - A_March 14, 2026_11.30.csv"),
        llm_jsonl=str(llm_dir / "gpt4.1_batch_evaluation_investment.jsonl"),
        prolific_path=str(BASE / "prolific_export_investment.csv"),
        exclude_rows=[],
        flag_consent_rows=[],
        step2_csv_path=str(
            BASE / "investment_export/Dialogue Quality Evaluation - Investment (B)_March 21, 2026_22.02.csv"
        ),
        step2_sample_ids=ANN2_INVEST_SAMPLES,
    )

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nReport → {args.out}")


if __name__ == "__main__":
    main()
