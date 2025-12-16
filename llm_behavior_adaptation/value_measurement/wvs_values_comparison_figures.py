import json
import math
import os
import re
from itertools import combinations
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from scipy import stats
from scipy.stats import friedmanchisquare, ttest_rel, wilcoxon

mpl.rcParams["font.family"] = "arial"


def compute_p_and_d_with_user_alignment(model_a, model_b):
    """Compute paired Wilcoxon p-value and Cohen's d on aligned users."""
    uid_a = {uid: ratio for uid, ratio in zip(model_a["user_ids"], model_a["ratios"])}
    uid_b = {uid: ratio for uid, ratio in zip(model_b["user_ids"], model_b["ratios"])}
    common_ids = list(set(uid_a) & set(uid_b))

    if not common_ids:
        return np.nan, np.nan

    x = np.array([uid_a[uid] for uid in common_ids])
    y = np.array([uid_b[uid] for uid in common_ids])

    try:
        p = stats.wilcoxon(x, y).pvalue
    except ValueError:
        p = np.nan

    diff = x - y
    std = np.std(diff, ddof=0)
    d = float(np.mean(diff) / std) if std != 0 else 0.0

    return p, d


def plot_pairwise_comparison_heatmap_aligned(model_ratios: dict):
    """
    Plots a heatmap showing pairwise p-values (upper) and Cohen's d (lower),
    ensuring user alignment per comparison.

    Args:
        model_ratios: dict of {model_name: {"user_ids": [...], "ratios": [...] }}
    """
    models = list(model_ratios.keys())
    n = len(models)

    pval_matrix = np.zeros((n, n), dtype=float)
    d_matrix = np.zeros((n, n), dtype=float)
    label_matrix = np.empty((n, n), dtype=object)

    for i in range(n):
        for j in range(n):
            if i == j:
                pval_matrix[i, j] = np.nan
                d_matrix[i, j] = np.nan
                label_matrix[i, j] = ""
            else:
                p, d = compute_p_and_d_with_user_alignment(model_ratios[models[i]], model_ratios[models[j]])
                pval_matrix[i, j] = p
                d_matrix[i, j] = d
                if i < j:
                    label_matrix[i, j] = f"p={p:.1e}" if not np.isnan(p) else ""
                else:
                    label_matrix[i, j] = f"d={d:.2f}" if not np.isnan(d) else ""

    fig, ax = plt.subplots(figsize=(1.2 * n, 1.0 * n))
    sns.heatmap(
        np.zeros_like(pval_matrix),
        annot=label_matrix,
        fmt="",
        cmap="Blues",
        xticklabels=models,
        yticklabels=models,
        cbar=False,
        linewidths=0.5,
        linecolor="gray",
        ax=ax,
    )
    plt.title("Pairwise Comparison: Wilcoxon p-values (↑) and Cohen’s d (↓)")
    plt.tight_layout()
    plt.show()


def compare_models_stats(ratio_dict):
    """
    Compare multiple models' per-user ratios against the best (lowest mean ratio).

    Returns a list of dicts with:
        - model name
        - mean ratio
        - std ratio
        - p-value (Wilcoxon vs best)
        - Cohen's d (vs best)

    Args:
        ratio_dict: dict of {model_name: list of per-user ratios}

    Returns:
        List[Dict[str, Any]] sorted by mean ratio (ascending)
    """
    mean_vals = {model: np.mean(ratios) for model, ratios in ratio_dict.items()}
    best_model = min(mean_vals, key=mean_vals.get)
    best_ratios = np.array(ratio_dict[best_model])

    results = []
    for model, ratios in ratio_dict.items():
        ratios = np.array(ratios)
        mean_val = float(np.mean(ratios))
        std_val = float(np.std(ratios, ddof=0))

        if model == best_model:
            p_val = None
            d_val = None
        else:
            # Paired Wilcoxon test
            stat, p = stats.wilcoxon(ratios, best_ratios)
            p_val = float(p)

            # Paired Cohen's d
            diffs = ratios - best_ratios
            denom = np.std(diffs, ddof=0)
            d_val = float(np.mean(diffs) / denom) if denom != 0 else 0.0

        results.append(
            {
                "model": model,
                "mean_ratio": mean_val,
                "std_ratio": std_val,
                "p_vs_best": p_val,
                "cohen_d_vs_best": d_val,
            }
        )

    # Sort by mean_ratio ascending
    return sorted(results, key=lambda x: x["mean_ratio"])


def plot_pairwise_map_only(
    datasets: List[Dict],
    model_names: Optional[List[str]] = None,
    *,
    metric_key: str = "ratios",  # "ratios" (recommended) or "divergences"
    pairwise_test: str = "ttest",  # "ttest" or "wilcoxon"
    correction: str = "holm",
    do_omnibus: bool = True,
    title: str = "Pairwise comparison map",
    output_dir: str = ".",
    prefix: str = "pairmap",
    dpi: int = 300,
):
    """
    Build a single pairwise comparison map for multiple models using per-user metrics.
    Upper triangle: -log10(Holm-corrected p) heatmap
    Lower triangle: Δ-median (row - col) with blue/red color overlay
    Diagonal: model medians
    """
    k = len(datasets)
    if k < 2:
        raise ValueError("Need at least two models.")

    if model_names is None:
        model_names = [f"M{i+1}" for i in range(k)]

    # 1) Align users across all models
    def get_ids(m):
        return m["user_ids"]

    common = set(get_ids(datasets[0]))
    for m in datasets[1:]:
        common &= set(get_ids(m))
    common = sorted(common)
    if not common:
        raise ValueError("No overlapping user_ids across models.")

    # 2) Build matrix X: rows=users, cols=models
    X = np.zeros((len(common), k), dtype=float)
    for j, m in enumerate(datasets):
        ids = m["user_ids"]
        vals = m[metric_key]
        if len(ids) != len(vals):
            raise ValueError(f"Model {j} has mismatched lengths for user_ids and {metric_key}.")
        idx_map = {u: i for i, u in enumerate(ids)}
        X[:, j] = [vals[idx_map[u]] for u in common]

    # 3) Summaries
    med = np.median(X, axis=0)

    # 4) Optional omnibus test (Friedman)
    omnibus = None
    if do_omnibus:
        stat, p = friedmanchisquare(*[X[:, i] for i in range(k)])
        omnibus = {"test": "Friedman", "stat": float(stat), "p": float(p)}

    # 5) Pairwise tests + Δ-median
    pairs = list(combinations(range(k), 2))
    p_raw, dmed = [], []
    for i, j in pairs:
        xi, xj = X[:, i], X[:, j]
        if pairwise_test == "ttest":
            _, p = ttest_rel(xi, xj)
        elif pairwise_test == "wilcoxon":
            try:
                _, p = wilcoxon(xi, xj, zero_method="zsplit", alternative="two-sided")
            except ValueError:
                _, p = ttest_rel(xi, xj)
        else:
            raise ValueError("pairwise_test must be 'ttest' or 'wilcoxon'")
        p_raw.append(p)
        dmed.append(float(np.median(xi) - np.median(xj)))  # lower = better (row better if negative)

    # Holm step-down correction
    if correction != "holm":
        raise ValueError("Only 'holm' correction is implemented.")
    m_tests = len(p_raw)
    order = np.argsort(p_raw)
    p_corr = np.empty_like(np.array(p_raw), dtype=float)
    for rank, idx in enumerate(order):
        p_corr[idx] = min((m_tests - rank) * p_raw[idx], 1.0)

    # 6) Matrices for plotting
    logp_mat = np.zeros((k, k), dtype=float)
    dmed_mat = np.zeros((k, k), dtype=float)
    for idx, (i, j) in enumerate(pairs):
        lp = -math.log10(max(p_corr[idx], 1e-300))
        logp_mat[i, j] = lp
        logp_mat[j, i] = lp
        dmed_mat[i, j] = dmed[idx]  # row - col
        dmed_mat[j, i] = -dmed[idx]

    # 7) Figure
    os.makedirs(output_dir, exist_ok=True)
    fig = plt.figure(figsize=(6.8, 6.2))
    plt.title(title)

    # Upper triangle heatmap: -log10 p
    im1 = plt.imshow(logp_mat)  # default colormap
    cb = plt.colorbar(im1, fraction=0.046, pad=0.04)
    cb.set_label("-log10(Holm-corrected p)")

    # Lower triangle color overlay for Δ-median using blue-white-red
    # Symmetric normalization around 0 using 95th percentile for stability
    lower_vals = dmed_mat[np.tril_indices(k, -1)]
    vmax = np.percentile(np.abs(lower_vals), 95) if lower_vals.size else 1.0
    vmax = max(vmax, 1e-6)
    norm = Normalize(vmin=-vmax, vmax=vmax)
    cmap = plt.get_cmap("bwr")  # blue (better) to red (worse)

    for i in range(k):
        for j in range(k):
            if i > j:
                color = cmap(norm(dmed_mat[i, j]))
                # semi-transparent rectangle
                plt.gca().add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color=color, alpha=0.45))

    # Ticks
    plt.xticks(range(k), model_names, rotation=30, ha="right")
    plt.yticks(range(k), model_names)

    # Text annotations
    for i in range(k):
        for j in range(k):
            if i == j:
                txt = f"med={med[i]:.3f}"
            elif i < j:
                p_disp = 10 ** (-logp_mat[i, j]) if logp_mat[i, j] > 0 else 1.0
                stars = "***" if p_disp < 1e-3 else ("**" if p_disp < 1e-2 else ("*" if p_disp < 5e-2 else ""))
                txt = f"p={p_disp:.1e}\n{stars}"
            else:
                txt = f"Δ={dmed_mat[i, j]:+.3f}"
            plt.text(j, i, txt, ha="center", va="center", fontsize=8)

    plt.tight_layout()
    png = os.path.join(output_dir, f"{prefix}_{metric_key}.png")
    pdf = os.path.join(output_dir, f"{prefix}_{metric_key}.pdf")
    plt.savefig(png, dpi=dpi)
    plt.savefig(pdf)
    plt.close(fig)

    # 8) Pack results
    pairwise = []
    for idx, (i, j) in enumerate(pairs):
        pairwise.append(
            {
                "i": i,
                "j": j,
                "name_i": model_names[i],
                "name_j": model_names[j],
                "p_raw": float(p_raw[idx]),
                "p_corr": float(p_corr[idx]),
                "delta_median": float(dmed[idx]),
            }
        )

    return {
        "aligned_user_count": len(common),
        "omnibus": omnibus,
        "pairwise": pairwise,
        "files": {"png": png, "pdf": pdf},
    }


def plot_user_divergence(data, baseline, formula="JSD", output_path=None):
    """
    Plots user divergence with error bars and a baseline, and optionally saves the plot.

    Parameters:
        data (list): A list of dictionaries containing group comparison data.
        baseline (float): The baseline value for average user divergence.
        output_path (str, optional): Path to save the plot as a PNG file. If None, the plot is not saved.
    """
    # Extract values
    groups = [item["compared_groups"] for item in data]
    avg_divergence = [item["compared_details"]["average_user_divergence"] for item in data]
    std_divergence = [item["compared_details"]["std_user_divergence"] for item in data]

    # Plot
    plt.figure(figsize=(12, 6))
    x = np.arange(len(groups))  # Numeric positions for groups
    plt.errorbar(
        x,
        avg_divergence,
        yerr=std_divergence,
        fmt="o",
        capsize=5,
        label="Average Divergence",
    )

    # Add baseline
    plt.axhline(y=baseline, color="red", linestyle="--", label=f"Baseline ({baseline:.3f})")

    # Customize plot
    plt.xticks(x, groups, rotation=45, ha="right")
    plt.xlabel("Compared Groups")
    plt.ylabel("Average User Divergence")
    plt.title(f"Average User Divergence ({formula}) with Baseline")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    # Save the plot if output_path is provided
    if output_path:
        plt.savefig(output_path, format="png", dpi=300)
        print(f"Plot saved to {output_path}")

    # Show the plot
    plt.show()


def plot_divergence_comparison_radar(
    datasets,
    baselines,
    labels,
    figsize=(16, 16),
    output_path=None,
    csv_path=None,
    cmap="tab20",
    *,
    pair_label_fontsize=None,
    label_radius_pad=0.01,  # Changed: Reduced from 0.05 to 0.01 for closer fit
    label_outline=True,
    group_to_abbrev=None,
    unknown_groups=("not sure",),
):
    """
    Optimized Radar chart.
    - Labels are attached closely to the outer line.
    """

    # ---- Validation ----
    if not (len(datasets) == len(baselines) == len(labels)):
        raise ValueError("All input lists must have equal length")

    # ---- Reorder models so 'Human' is first ----
    if "Human" in labels:
        h_idx = labels.index("Human")
        order = [h_idx] + [i for i in range(len(labels)) if i != h_idx]
        labels = [labels[i] for i in order]
        baselines = [baselines[i] for i in order]
        datasets = [datasets[i] for i in order]

    # ---- Baselines & model count ----
    baselines = [b["overall_baseline"] for b in baselines]
    n_models = len(datasets)
    unknown_set = {u.strip().lower() for u in (unknown_groups or ())}

    # ---- Helper functions ----
    def _canon_pair_full(s: str) -> str:
        return s.replace("--", " vs ").strip()

    def _split_pair(pair_full: str):
        if " vs " in pair_full:
            l, r = pair_full.split(" vs ", 1)
            return l.strip(), r.strip()
        return pair_full.strip(), None

    def _is_unknown_pair(pair_full: str) -> bool:
        l, r = _split_pair(pair_full)
        if (l and l.lower() in unknown_set) or (r and r.lower() in unknown_set):
            return True
        return False

    def _pair_to_abbrev(pair_full: str) -> str:
        if not group_to_abbrev:
            return pair_full
        l, r = _split_pair(pair_full)
        if r is None:
            return group_to_abbrev.get(l, l)
        la = group_to_abbrev.get(l, l)
        ra = group_to_abbrev.get(r, r)
        return f"{la} vs {ra}"

    # ---- Collect unique FULL group pairs ----
    all_pairs_full = set()
    for data in datasets:
        for item in data:
            g_full = _canon_pair_full(item["compared_groups"])
            if _is_unknown_pair(g_full):
                continue
            all_pairs_full.add(g_full)

    sorted_pairs_full = sorted(all_pairs_full)
    n_groups = len(sorted_pairs_full)

    if n_groups == 0:
        print("Warning: No valid groups found to plot.")
        return None, None

    # ---- Values (ratios) ----
    values = np.zeros((n_models, n_groups))
    for mi, data in enumerate(datasets):
        base = baselines[mi]
        for item in data:
            g_full = _canon_pair_full(item["compared_groups"])
            if _is_unknown_pair(g_full):
                continue
            j = sorted_pairs_full.index(g_full)
            div = float(item["compared_details"]["average_divergence"])
            values[mi, j] = div / base if base != 0 else np.nan

    display_pairs = [_pair_to_abbrev(p) for p in sorted_pairs_full]

    # ---- CSV export ----
    if csv_path:
        data_dict = {"Group_Full": sorted_pairs_full, "Group": display_pairs}
        for mi in range(n_models):
            data_dict[labels[mi]] = values[mi, :]
        pd.DataFrame(data_dict).to_csv(csv_path, index=False)

    # ---- Radar scaffold ----
    angles = np.linspace(0, 2 * np.pi, n_groups, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    cmap_obj = get_cmap(cmap)
    colors = [cmap_obj(i % max(1, getattr(cmap_obj, "N", 20))) for i in range(n_models)]

    # ---- Draw series ----
    human_present = len(labels) > 0 and labels[0] == "Human"
    start_idx = 1 if human_present else 0
    for mi in range(start_idx, n_models):
        row = list(values[mi]) + [values[mi][0]]
        ax.plot(angles, row, color=colors[mi], linewidth=1.8, label=labels[mi], zorder=3)
        ax.fill(angles, row, color=colors[mi], alpha=0.18, zorder=2)

    if human_present:
        hrow = list(values[0]) + [values[0][0]]
        ax.plot(angles, hrow, color=colors[0], linewidth=3.2, label=labels[0], zorder=10)
        ax.scatter(angles[:-1], values[0], color=colors[0], s=45, zorder=11)
        ax.fill(angles, hrow, color=colors[0], alpha=0.35, zorder=5)

    # ---- Limits ----
    max_val = np.nanmax(values)
    if not np.isfinite(max_val) or max_val <= 0:
        max_val = 1.0

    limit_val = max_val * 1.02
    ax.set_ylim(0, limit_val)

    yticks = np.linspace(0, max_val, num=5)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{y:.2f}" if y < 1 else f"{y:.1f}" for y in yticks], fontsize=12)

    # ---- Font & Labels ----
    if pair_label_fontsize is None:
        # Changed: Significantly boosted base font size and minimum floor
        # Was: max(16, 34 - ...), Now: max(22, 45 - ...)
        pair_label_fontsize = max(22, int(45 - 0.8 * n_groups))

    ax.set_xticks([])

    # Changed: Apply the negative padding to pull text inside/onto the line
    r_label_pos = limit_val * (1.0 + label_radius_pad)

    # Changed: Thicker white stroke (4) to ensure legibility when overlapping lines
    effects = [pe.withStroke(linewidth=4, foreground="white")] if label_outline else None

    for angle, txt in zip(angles[:-1], display_pairs):
        a = (angle + np.pi) % (2 * np.pi) - np.pi
        ha = "left" if (-np.pi / 2 < a < np.pi / 2) else ("center" if abs(a) == np.pi / 2 else "right")

        # Dynamic VA to ensure the text "hugs" the line
        if abs(a) < np.pi / 8:  # Top
            va = "bottom"
        elif abs(a) > 7 * np.pi / 8:  # Bottom
            va = "top"
        else:  # Sides
            va = "center"

        ax.text(
            angle,
            r_label_pos,
            txt,
            ha=ha,
            va=va,
            fontsize=pair_label_fontsize,
            path_effects=effects,
            weight="bold",
            clip_on=False,
        )

    ax.spines["polar"].set_visible(False)
    ax.grid(color="#AAAAAA", linestyle="--", alpha=0.7)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, bbox_inches="tight", dpi=300)

    # ---- Legend ----
    leg_handles, leg_labels = ax.get_legend_handles_labels()
    handle_by_label = {lab: h for h, lab in zip(leg_handles, leg_labels)}
    ordered_handles = [handle_by_label[lab] for lab in labels if lab in handle_by_label]
    legend_fig, legend_ax = plt.subplots(figsize=(14, 1))
    legend_ax.axis("off")
    legend_ax.legend(
        ordered_handles,
        labels,
        loc="center",
        fontsize=30,
        frameon=False,
        ncol=min(7, n_models),
    )
    if output_path:
        legend_fig.savefig(
            output_path.replace(".pdf", "_legend.pdf").replace(".png", "_legend.png"),
            bbox_inches="tight",
            dpi=300,
        )

    return fig, ax


def plot_divergence_comparison_heatmap(
    *,
    datasets: Sequence[Sequence[Dict]],
    baselines: Sequence[Dict],
    labels: Sequence[str],
    figsize: Tuple[int, int] = (12, 6),
    output_path: Optional[str] = None,
    csv_path: Optional[str] = None,
    cmap: str = "viridis",
    darker_is_larger: bool = False,
    emphasize_label: Optional[str] = "Human",
    # ---- General ordering controls ----
    sort_by_defined_order: bool = True,
    defined_order: Optional[Sequence[str]] = None,
    pair_normalizer: Optional[Callable[[str], Tuple[str, str]]] = None,
    order_pairs: Optional[Callable[[List[str]], List[str]]] = None,  # full custom column order
    # ---- Presentation ----
    annotate: bool = True,  # show numbers in cells by default
    annotate_fontsize: int = 9,
    fmt: str = ".2f",  # number formatting (2 decimals)
    grid: bool = True,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> pd.DataFrame:
    """
    General heatmap for divergence comparisons (ratio over baseline) across models.

    Inputs mirror your radar function:
      - datasets: list of per-model lists. Each item is a dict like:
          {"compared_groups": "A--B" or "A vs B",
           "compared_details": {"average_divergence": float}}
      - baselines: list of dicts with key "overall_baseline" (one per model)
      - labels: list of model names

    Ordering & parsing:
      - defined_order: optional sequence defining left/right token order for sorting pairs
      - sort_by_defined_order: if True, columns sorted by left then right token
      - pair_normalizer: optional function(raw_str) -> (left_token, right_token)
      - order_pairs: optional function that returns a custom ordered list of pairs (overrides defined_order)

    Display:
      - darker_is_larger: reverse colormap so larger = darker
      - emphasize_label: keep this row first and draw an outline
      - annotate: write values in cells using `fmt` (default '.2f') with auto-contrast text
      - grid: dotted minor gridlines
      - vmin/vmax: fix color scale; None = auto min/max

    Returns:
      - DataFrame of ratios (rows=models, columns=group pairs).
    """
    if not (len(datasets) == len(baselines) == len(labels)):
        raise ValueError("datasets, baselines, and labels must have equal length")

    # ---- Pair parsing & normalization ----
    _pair_re = re.compile(r"\s*(.*?)\s*(?:vs|--)\s*(.*?)\s*$", flags=re.IGNORECASE)

    def _default_normalizer(raw: str) -> Tuple[str, str]:
        m = _pair_re.match(raw)
        if not m:
            s = raw.strip()
            return (s, s)
        a, b = m.group(1).strip(), m.group(2).strip()
        return a, b

    norm = pair_normalizer or _default_normalizer

    def _canon_pair(raw: str) -> str:
        a, b = norm(raw.replace("--", " vs "))
        return f"{a} vs {b}"

    # ---- Gather pairs ----
    all_pairs = set()
    for data in datasets:
        for item in data:
            p = _canon_pair(item["compared_groups"])
            if "unknown" not in p.lower():
                all_pairs.add(p)
    pairs = sorted(all_pairs)  # provisional

    # ---- Column ordering ----
    if order_pairs is not None:
        pairs = order_pairs(pairs)
    elif sort_by_defined_order:

        def token_rank(tok: str) -> Tuple[int, str]:
            if defined_order is None:
                return (0, tok)  # alpha fallback
            try:
                return (defined_order.index(tok), tok)
            except ValueError:
                return (len(defined_order), tok)

        def _pair_key(p: str) -> Tuple[Tuple[int, str], Tuple[int, str], str]:
            a, b = norm(p)
            return (token_rank(a), token_rank(b), p)

        pairs = sorted(pairs, key=_pair_key)

    # ---- Build ratio matrix ----
    n_models, n_cols = len(datasets), len(pairs)
    ratios = np.zeros((n_models, n_cols), dtype=float)
    base_vals = [float(b["overall_baseline"]) for b in baselines]
    p2idx = {p: j for j, p in enumerate(pairs)}

    for mi, data in enumerate(datasets):
        base = base_vals[mi]
        for item in data:
            p = _canon_pair(item["compared_groups"])
            if p in p2idx:
                j = p2idx[p]
                div = float(item["compared_details"]["average_divergence"])
                ratios[mi, j] = div / base if base != 0 else np.nan

    mat = pd.DataFrame(ratios, index=list(labels), columns=pairs)

    # Emphasize row (e.g., Human)
    if emphasize_label in mat.index:
        mat = mat.loc[[emphasize_label] + [r for r in mat.index if r != emphasize_label], :]

    # CSV export
    if csv_path:
        out = pd.DataFrame({"Group": mat.columns})
        for r in mat.index:
            out[r] = mat.loc[r].to_numpy()
        out.to_csv(csv_path, index=False)

    # Colormap & scaling
    cm = get_cmap(cmap)
    if darker_is_larger:
        try:
            cm = get_cmap(cmap + "_r")
        except ValueError:
            cm = cm.reversed()
    if vmin is None:
        vmin = float(np.nanmin(mat.values))
    if vmax is None:
        vmax = float(np.nanmax(mat.values))

    # ---- Plot ----
    plt.figure(figsize=figsize)
    im = plt.imshow(mat.values, aspect="auto", cmap=cm, vmin=vmin, vmax=vmax)

    plt.yticks(range(mat.shape[0]), mat.index)
    plt.xticks(range(mat.shape[1]), mat.columns, rotation=45, ha="right")

    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)  # noqa F841
    # cbar.set_label("Ratio over baseline", rotation=90)

    if emphasize_label in mat.index:
        r = list(mat.index).index(emphasize_label)
        plt.gca().add_patch(Rectangle((-0.5, r - 0.5), mat.shape[1], 1, fill=False, lw=2))

    if grid:
        ax = plt.gca()
        ax.set_xticks(np.arange(-0.5, mat.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, mat.shape[0], 1), minor=True)
        ax.grid(which="minor", linestyle=":", linewidth=0.5)

    # ---- Inline annotations (2 decimals by default) ----
    if annotate:
        arr = mat.values
        # Normalize for auto-contrast
        norm_arr = (arr - vmin) / (vmax - vmin + 1e-12)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                val = arr[i, j]
                if np.isnan(val):
                    txt = "NaN"
                else:
                    txt = format(val, fmt)
                # light text on dark cells, dark text on light cells
                color = "white" if norm_arr[i, j] > 0.6 else "black"
                plt.text(
                    j,
                    i,
                    txt,
                    ha="center",
                    va="center",
                    fontsize=annotate_fontsize,
                    color=color,
                )

    # plt.title("Divergence Comparison — Heatmap (ratio over baseline)")
    # plt.xlabel("Group pair")
    # plt.ylabel("Model")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, bbox_inches="tight", dpi=300)

    return mat


def display_comparison(
    model_list,
    scenario: str = "ba_user",
    attribute: str = "age",
    cmap="tab20",
    specific_name=None,
    pair_label_fontsize=None,
    group_to_abbrev=None,  # e.g., {"Lower class": "LC", ...}
    unknown_groups=("not sure",),
):
    datasets = []
    baselines = []
    for model_label in model_list:
        try:
            with open(
                f"wvs_values_results/{model_label}/experiments_results.json",
                "r",
                encoding="utf-8",
            ) as jl_file:
                if model_label.lower() == "human":
                    experiments_results = json.load(jl_file)[attribute]
                else:
                    experiments_results = json.load(jl_file)[f"{scenario}_results"][attribute]
                datasets.append(experiments_results["group_distances"])
                baselines.append(experiments_results["baseline"])
        except Exception as e:
            print(model_label)
            print(str(e))

    os.makedirs(f"wvs_images/{scenario}/", exist_ok=True)

    output_path = (
        f"wvs_images/{scenario}/{specific_name}.pdf"
        if specific_name is not None
        else f"wvs_images/{scenario}/{attribute}.pdf"
    )
    csv_path = (
        f"wvs_images/{scenario}/{specific_name}.csv"
        if specific_name is not None
        else f"wvs_images/{scenario}/{attribute}.csv"
    )

    plot_divergence_comparison_radar(
        datasets=datasets,
        baselines=baselines,
        labels=model_list,
        output_path=output_path,
        csv_path=csv_path,
        cmap=cmap,
        pair_label_fontsize=pair_label_fontsize,
        group_to_abbrev=group_to_abbrev,
        unknown_groups=unknown_groups,
    )


def display_comparison_heatmap(
    model_list,
    scenario: str = "ba_user",
    attribute: str = "age",
    cmap="tab20",
    specific_name=None,
    defined_order=None,
):
    """For generating heatmap figure

    Args:
        model_list (_type_): _description_
        scenario (str, optional): _description_. Defaults to "ba_user".
        attribute (str, optional): _description_. Defaults to "age".
        cmap (str, optional): _description_. Defaults to "tab20".
        specific_name (_type_, optional): _description_. Defaults to None.
        defined_order (_type_, optional): _description_. Defaults to None.
    """
    datasets = []
    baselines = []
    for model_label in model_list:
        try:
            with open(
                f"wvs_values_results/{model_label}/experiments_results.json",
                "r",
                encoding="utf-8",
            ) as jl_file:
                if model_label.lower() == "human":
                    experiments_results = json.load(jl_file)[attribute]
                else:
                    experiments_results = json.load(jl_file)[f"{scenario}_results"][attribute]
                datasets.append(experiments_results["group_distances"])
                baselines.append(experiments_results["baseline"])
        except Exception as e:
            print(model_label)
            print(str(e))

    os.makedirs(f"wvs_images/{scenario}/", exist_ok=True)

    output_path = (
        f"wvs_images/{scenario}/{specific_name}_heatmap.pdf"
        if specific_name is not None
        else f"wvs_images/{scenario}/{attribute}_heatmap.pdf"
    )
    # csv_path = (
    #     f"wvs_images/{scenario}/{specific_name}.csv"
    #     if specific_name is not None
    #     else f"wvs_images/{scenario}/{attribute}.csv"
    # )

    plot_divergence_comparison_heatmap(
        datasets=datasets,
        baselines=baselines,
        labels=model_list,
        cmap=cmap,
        darker_is_larger=True,
        emphasize_label="Human",
        sort_by_defined_order=True,
        defined_order=defined_order,
        output_path=output_path,
    )


def display_model_consistency_comparison(model_list, dialogue_topic: str = "career"):
    """For generating heatmap figure

    Args:
        model_list (_type_): _description_
    """
    ratio_dict = {}
    for model_label in model_list:
        try:
            with open(
                f"wvs_values_results/{model_label}/experiments_results.json",
                "r",
                encoding="utf-8",
            ) as jl_file:
                experiments_results = json.load(jl_file)["cross_datasets_results"][dialogue_topic]
                ratio_dict[model_label] = experiments_results["per_user"]["ratios"]
        except Exception as e:
            print(model_label)
            print(str(e))

    model_comparison_stats = compare_models_stats(ratio_dict=ratio_dict)

    os.makedirs("wvs_images/consistency_comparison/", exist_ok=True)

    with open(
        f"wvs_images/consistency_comparison/{dialogue_topic}_comparison_results.json",
        "w",
        encoding="utf-8",
    ) as c_f:
        json.dump(model_comparison_stats, c_f, indent=2)

    # plot_pairwise_comparison_heatmap_aligned(ratio_dict)

    # output_path = f"wvs_images/consistency_comparison/{dialogue_topic}_heatmap.pdf"

    # stats = plot_pairwise_map_only(
    #     datasets=datasets,
    #     model_names=model_list,
    #     output_dir="wvs_images/consistency_comparison/",
    # )

    # print(stats["omnibus"])

    # plot_divergence_comparison_heatmap(
    #     datasets=datasets,
    #     baselines=baselines,
    #     labels=model_list,
    #     cmap=cmap,
    #     darker_is_larger=True,
    #     emphasize_label="Human",
    #     sort_by_defined_order=True,
    #     defined_order=defined_order,
    #     output_path=output_path,
    # )


edu_group_to_abbrev = {
    "Basic education": "BE",
    "High school & equivalent": "HS",
    "Short-cycle tertiary": "SCT",
    "Bachelor": "BA",
    "Master’s & Doctoral": "MD",
}

occupation_group_to_abbrev = {
    "Clerical & Sales": "CS",
    "Skilled & Semi-Skilled": "SS",
    "Service & Labor": "SL",
    "Managerial / Professional": "MP",
    "Agricultural Related": "AG",
    "Unemployed / No Job": "NJ",
}

socioeconomic_group_to_abbrev_class = {
    "Lower class": "LC",
    "Working class": "WC",
    "Lower middle class": "LMC",
    "Upper middle class": "UMC",
    "Upper class": "UC",
    # "not sure" intentionally omitted; it will be filtered by unknown_groups
}

# display_model_consistency_comparison(
#     [
#         "Llama-3.1-8B-Instruct",
#         "Llama-3.1-70B-Instruct",
#         "DeepSeek-V3",
#         "Qwen2.5-7B-Instruct",
#         "Qwen2.5-72B-Instruct",
#         "QwQ-32B",
#     ],
#     dialogue_topic="career",
# )

# display_model_consistency_comparison(
#     [
#         "Llama-3.1-8B-Instruct",
#         "Llama-3.1-70B-Instruct",
#         "DeepSeek-V3",
#         "Qwen2.5-7B-Instruct",
#         "Qwen2.5-72B-Instruct",
#         "QwQ-32B",
#     ],
#     dialogue_topic="investment",
# )

# BA User
display_comparison(
    [
        "Human",
        "Llama-3.1-8B-Instruct",
        "Llama-3.1-70B-Instruct",
        "DeepSeek-V3",
        "Qwen2.5-7B-Instruct",
        "Qwen2.5-72B-Instruct",
        "QwQ-32B",
    ],
    cmap="tab10",
    attribute="highest_level_of_education",
    scenario="ba_user",
    specific_name="ba_user_education_radar",
    # pair_label_fontsize=14,
    group_to_abbrev=edu_group_to_abbrev,
)

display_comparison(
    [
        "Human",
        "Llama-3.1-8B-Instruct",
        "Llama-3.1-70B-Instruct",
        "DeepSeek-V3",
        "Qwen2.5-7B-Instruct",
        "Qwen2.5-72B-Instruct",
        "QwQ-32B",
    ],
    cmap="tab10",
    attribute="age",
    scenario="ba_user",
    specific_name="ba_user_age_radar",
    # pair_label_fontsize=14,
)


display_comparison(
    [
        "Human",
        "Llama-3.1-8B-Instruct",
        "Llama-3.1-70B-Instruct",
        "DeepSeek-V3",
        "Qwen2.5-7B-Instruct",
        "Qwen2.5-72B-Instruct",
        "QwQ-32B",
    ],
    cmap="tab10",
    attribute="socioeconomic_status",
    scenario="ba_user",
    specific_name="ba_user_socioeconomic_status_radar",
    # pair_label_fontsize=14,
    group_to_abbrev=socioeconomic_group_to_abbrev_class,
)

display_comparison(
    [
        "Human",
        "Llama-3.1-8B-Instruct",
        "Llama-3.1-70B-Instruct",
        "DeepSeek-V3",
        "Qwen2.5-7B-Instruct",
        "Qwen2.5-72B-Instruct",
        "QwQ-32B",
    ],
    cmap="tab10",
    attribute="occupation_group",
    scenario="ba_user",
    specific_name="ba_user_occupation_group_radar",
    # pair_label_fontsize=14,
    group_to_abbrev=occupation_group_to_abbrev,
)

# Career Dialogue
display_comparison(
    [
        "Human",
        "Llama-3.1-8B-Instruct",
        "Llama-3.1-70B-Instruct",
        "DeepSeek-V3",
        "Qwen2.5-7B-Instruct",
        "Qwen2.5-72B-Instruct",
        "QwQ-32B",
    ],
    cmap="tab10",
    attribute="highest_level_of_education",
    scenario="ba_dialogue_career",
    specific_name="ba_dialogue_career_education_radar",
    # pair_label_fontsize=14,
    group_to_abbrev=edu_group_to_abbrev,
)

display_comparison(
    [
        "Human",
        "Llama-3.1-8B-Instruct",
        "Llama-3.1-70B-Instruct",
        "DeepSeek-V3",
        "Qwen2.5-7B-Instruct",
        "Qwen2.5-72B-Instruct",
        "QwQ-32B",
    ],
    cmap="tab10",
    attribute="age",
    scenario="ba_dialogue_career",
    specific_name="ba_dialogue_career_age_radar",
    # pair_label_fontsize=14,
)


display_comparison(
    [
        "Human",
        "Llama-3.1-8B-Instruct",
        "Llama-3.1-70B-Instruct",
        "DeepSeek-V3",
        "Qwen2.5-7B-Instruct",
        "Qwen2.5-72B-Instruct",
        "QwQ-32B",
    ],
    cmap="tab10",
    attribute="socioeconomic_status",
    scenario="ba_dialogue_career",
    specific_name="ba_dialogue_career_socioeconomic_status_radar",
    # pair_label_fontsize=14,
    group_to_abbrev=socioeconomic_group_to_abbrev_class,
)


display_comparison(
    [
        "Human",
        "Llama-3.1-8B-Instruct",
        "Llama-3.1-70B-Instruct",
        "DeepSeek-V3",
        "Qwen2.5-7B-Instruct",
        "Qwen2.5-72B-Instruct",
        "QwQ-32B",
    ],
    cmap="tab10",
    attribute="occupation_group",
    scenario="ba_dialogue_career",
    specific_name="ba_dialogue_career_occupation_group_radar",
    # pair_label_fontsize=14,
    group_to_abbrev=occupation_group_to_abbrev,
)

# Investment Dialogue
display_comparison(
    [
        "Human",
        "Llama-3.1-8B-Instruct",
        "Llama-3.1-70B-Instruct",
        "DeepSeek-V3",
        "Qwen2.5-7B-Instruct",
        "Qwen2.5-72B-Instruct",
        "QwQ-32B",
    ],
    cmap="tab10",
    attribute="highest_level_of_education",
    scenario="ba_dialogue_investment",
    specific_name="ba_dialogue_investment_education_radar",
    # pair_label_fontsize=14,
    group_to_abbrev=edu_group_to_abbrev,
)

display_comparison(
    [
        "Human",
        "Llama-3.1-8B-Instruct",
        "Llama-3.1-70B-Instruct",
        "DeepSeek-V3",
        "Qwen2.5-7B-Instruct",
        "Qwen2.5-72B-Instruct",
        "QwQ-32B",
    ],
    cmap="tab10",
    attribute="age",
    scenario="ba_dialogue_investment",
    specific_name="ba_dialogue_investment_age_radar",
    # pair_label_fontsize=14,
)


display_comparison(
    [
        "Human",
        "Llama-3.1-8B-Instruct",
        "Llama-3.1-70B-Instruct",
        "DeepSeek-V3",
        "Qwen2.5-7B-Instruct",
        "Qwen2.5-72B-Instruct",
        "QwQ-32B",
    ],
    cmap="tab10",
    attribute="socioeconomic_status",
    scenario="ba_dialogue_investment",
    specific_name="ba_dialogue_investment_socioeconomic_status_radar",
    # pair_label_fontsize=14,
    group_to_abbrev=socioeconomic_group_to_abbrev_class,
)

display_comparison(
    [
        "Human",
        "Llama-3.1-8B-Instruct",
        "Llama-3.1-70B-Instruct",
        "DeepSeek-V3",
        "Qwen2.5-7B-Instruct",
        "Qwen2.5-72B-Instruct",
        "QwQ-32B",
    ],
    cmap="tab10",
    attribute="occupation_group",
    scenario="ba_dialogue_investment",
    specific_name="ba_dialogue_investment_occupation_group_radar",
    # pair_label_fontsize=14,
    group_to_abbrev=occupation_group_to_abbrev,
)

# display_comparison_heatmap(
#     [
#         "Human",
#         "Llama-3.1-8B-Instruct",
#         "Llama-3.1-70B-Instruct",
#         "DeepSeek-V3",
#         "Qwen2.5-7B-Instruct",
#         "Qwen2.5-72B-Instruct",
#         "QwQ-32B",
#     ],
#     cmap="viridis",
#     attribute="age",
#     scenario="ba_user",
#     specific_name="ba_user_age_radar",
#     defined_order=["<30", "30-40", "40-50", "50-60", ">60"],
# )


# display_comparison_heatmap(
#     [
#         "Human",
#         "Llama-3.1-8B-Instruct",
#         "Llama-3.1-70B-Instruct",
#         "DeepSeek-V3",
#         "Qwen2.5-7B-Instruct",
#         "Qwen2.5-72B-Instruct",
#         # "QwQ-32B",
#     ],
#     cmap="viridis",
#     attribute="highest_level_of_education",
#     scenario="ba_user",
#     # extra_rules=["<30", ">60"],
#     specific_name="ba_user_education",
#     defined_order=[
#         "Basic education",
#         "High school & equivalent",
#         "Short-cycle tertiary",
#         "Bachelor",
#         "Master’s & Doctoral",
#     ],
# )

# display_comparison_heatmap(
#     [
#         "Human",
#         "Llama-3.1-8B-Instruct",
#         "Llama-3.1-70B-Instruct",
#         "DeepSeek-V3",
#         "Qwen2.5-7B-Instruct",
#         "Qwen2.5-72B-Instruct",
#         # "QwQ-32B",
#     ],
#     cmap="viridis",
#     attribute="socioeconomic_status",
#     scenario="ba_user",
#     # extra_rules=["<30", ">60"],
#     specific_name="ba_user_socioeconomic_status",
#     defined_order=[
#         "Lower class",
#         "Working class",
#         "Lower middle class",
#         "Upper middle class",
#         "Upper class",
#     ],
# )

# display_comparison(
#     [
#         "Llama3.1-8B-Instruct",
#         "Llama3.1-70B-Instruct",
#         "DeepSeek-V3",
#         "Qwen2.5-7B-Instruct",
#         "Qwen2.5-72B-Instruct",
#         "QwQ-32B",
#     ],
#     group_spacing=1.75,
#     cmap="tab10",
#     attribute="age",
#     scenario="BA_user",
#     # extra_rules=["<30", ">60"],
#     specific_name="BA_user_age_radar",
# )

# display_comparison(
#     [
#         "Llama3.1-8B-Instruct",
#         "Llama3.1-70B-Instruct",
#         "DeepSeek-V3",
#         "Qwen2.5-7B-Instruct",
#         "Qwen2.5-72B-Instruct",
#         "QwQ-32B",
#     ],
#     group_spacing=1.75,
#     cmap="tab10",
#     attribute="education",
#     scenario="BA_dialogue",
#     specific_name="BA_dialogue_education_radar",
# )

# display_comparison(
#     [
#         "Llama3.1-8B-Instruct",
#         "Llama3.1-70B-Instruct",
#         "DeepSeek-V3",
#         "Qwen2.5-7B-Instruct",
#         "Qwen2.5-72B-Instruct",
#         "QwQ-32B",
#     ],
#     group_spacing=1.75,
#     cmap="tab10",
#     attribute="education",
#     scenario="BA_user",
#     specific_name="BA_user_education_radar",
# )


# display_comparison(
#     [
#         "Llama3.1-8B-Instruct",
#         "Llama3.1-70B-Instruct",
#         "DeepSeek-V3",
#         "Qwen2.5-7B-Instruct",
#         "Qwen2.5-72B-Instruct",
#         "QwQ-32B",
#     ],
#     group_spacing=1.5,
#     cmap="tab10",
#     attribute="development_level",
#     scenario="BA_dialogue",
#     specific_name="BA_dialogue_development_level_radar",
# )
# display_comparison(
#     [
#         "Llama3.1-8B-Instruct",
#         "Llama3.1-70B-Instruct",
#         "DeepSeek-V3",
#         "Qwen2.5-7B-Instruct",
#         "Qwen2.5-72B-Instruct",
#         "QwQ-32B",
#     ],
#     group_spacing=1.5,
#     cmap="tab10",
#     attribute="development_level",
#     scenario="BA_user",
#     specific_name="BA_user_development_level_radar",
# )

# display_comparison(
#     [
#         "Llama3.1-8B-Instruct",
#         "Llama3.1-70B-Instruct",
#         "DeepSeek-V3",
#         "Qwen2.5-7B-Instruct",
#         "Qwen2.5-72B-Instruct",
#         "QwQ-32B",
#     ],
#     group_spacing=1.5,
#     cmap="tab10",
#     attribute="position_level",
#     scenario="BA_user",
#     specific_name="BA_user_position_level_radar",
# )

# display_comparison(
#     [
#         "Llama3.1-8B-Instruct",
#         "Llama3.1-70B-Instruct",
#         "DeepSeek-V3",
#         "Qwen2.5-7B-Instruct",
#         "Qwen2.5-72B-Instruct",
#         "QwQ-32B",
#     ],
#     group_spacing=1.5,
#     cmap="tab10",
#     attribute="position_level",
#     scenario="BA_dialogue",
#     specific_name="BA_dialogue_position_level_radar",
# )

# display_comparison(
#     [
#         "Llama3.1-8B-Instruct",
#         "Llama3.1-70B-Instruct",
#         "DeepSeek-V3",
#         "Qwen2.5-7B-Instruct",
#         "Qwen2.5-72B-Instruct",
#         "QwQ-32B",
#     ],
#     group_spacing=1.75,
#     cmap="tab10",
#     attribute="job_category",
#     scenario="BA_user",
#     specific_name="BA_user_job_category_radar",
#     # extra_rules=["Business", "Science"]
# )

# display_comparison(
#     [
#         "Llama3.1-8B-Instruct",
#         "Llama3.1-70B-Instruct",
#         "DeepSeek-V3",
#         "Qwen2.5-7B-Instruct",
#         "Qwen2.5-72B-Instruct",
#         "QwQ-32B",
#     ],
#     group_spacing=1.75,
#     cmap="tab10",
#     attribute="job_category",
#     scenario="BA_dialogue",
#     specific_name="BA_dialogue_job_category_radar",
# )
