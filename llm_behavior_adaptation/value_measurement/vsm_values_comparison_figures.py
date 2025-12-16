import json
import os
import re
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.cm import get_cmap
from matplotlib.patches import Rectangle
from scipy import stats

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
    mean_vals = {model: np.nanmean(ratios) for model, ratios in ratio_dict.items()}
    print(mean_vals)
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
            d_val = float(np.nanmean(diffs) / denom) if denom != 0 else 0.0

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


def plot_divergence_comparison_radar(
    datasets,
    baselines,
    labels,
    figsize=(16, 16),
    output_path=None,
    csv_path=None,
    cmap="tab20",
    label_pad=-5,  # kept for backward compatibility (no effect now)
    *,
    pair_label_fontsize=None,  # if None: auto
    label_radius_pad=0.12,  # how far beyond max radius to place labels (fraction)
    label_outline=True,  # add white outline to label text
    group_to_abbrev=None,  # e.g., {"Lower class": "LC", ...}
    unknown_groups=("not sure",),  # filter these (case-insensitive) at draw time
):
    """
    Radar chart for ratio-over-baseline divergences.
    - Inputs are reordered so 'Human' is first (if present).
    - Keeps original groups (no merging).
    - Filters pairs that include any label in `unknown_groups`.
    - Uses abbreviations for spoke labels if `group_to_abbrev` provided.
    """

    # ---- Validation ----
    if not (len(datasets) == len(baselines) == len(labels)):
        raise ValueError("All input lists must have equal length")

    # ---- Reorder models so 'Human' is first (others keep relative order) ----
    if "Human" in labels:
        h_idx = labels.index("Human")
        order = [h_idx] + [i for i in range(len(labels)) if i != h_idx]
        labels = [labels[i] for i in order]
        baselines = [baselines[i] for i in order]
        datasets = [datasets[i] for i in order]
    # else: keep as-is

    # ---- Baselines & model count ----
    baselines = [b["overall_baseline"] for b in baselines]
    n_models = len(datasets)

    # Prepare unknown set for quick match
    unknown_set = {u.strip().lower() for u in (unknown_groups or ())}

    def _canon_pair_full(s: str) -> str:
        """Normalize to 'A vs B' (full names)."""
        return s.replace("--", " vs ").strip()

    def _split_pair(pair_full: str):
        """Return (left, right) full labels; right can be None if no separator."""
        if " vs " in pair_full:
            l, r = pair_full.split(" vs ", 1)
            return l.strip(), r.strip()
        return pair_full.strip(), None

    def _is_unknown_pair(pair_full: str) -> bool:
        """True if any side is in unknown_set (case-insensitive)."""
        l, r = _split_pair(pair_full)
        if l and l.lower() in unknown_set:
            return True
        if r and r.lower() in unknown_set:
            return True
        return False

    def _pair_to_abbrev(pair_full: str) -> str:
        """Convert 'A vs B' -> 'ABBR(A) vs ABBR(B)' if mapping provided."""
        if not group_to_abbrev:
            return pair_full
        l, r = _split_pair(pair_full)
        if r is None:
            return group_to_abbrev.get(l, l)
        la = group_to_abbrev.get(l, l)
        ra = group_to_abbrev.get(r, r)
        return f"{la} vs {ra}"

    # ---- Collect unique FULL group pairs (skip unknown) ----
    all_pairs_full = set()
    for data in datasets:
        for item in data:
            g_full = _canon_pair_full(item["compared_groups"])
            if _is_unknown_pair(g_full):
                continue
            all_pairs_full.add(g_full)

    sorted_pairs_full = sorted(all_pairs_full)
    n_groups = len(sorted_pairs_full)

    # ---- Values (ratios), indexed by FULL pairs ----
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

    # ---- Display labels (abbrev if given) ----
    display_pairs = [_pair_to_abbrev(p) for p in sorted_pairs_full]

    # ---- CSV export (Human first in columns) ----
    if csv_path:
        data_dict = {"Group_Full": sorted_pairs_full, "Group": display_pairs}
        for mi in range(n_models):
            data_dict[labels[mi]] = values[mi, :]
        pd.DataFrame(data_dict).to_csv(csv_path, index=False)

    # ---- Radar scaffold ----
    angles = np.linspace(0, 2 * np.pi, n_groups, endpoint=False).tolist()
    angles += angles[:1]  # close

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    cmap_obj = get_cmap(cmap)
    colors = [cmap_obj(i % max(1, getattr(cmap_obj, "N", 20))) for i in range(n_models)]

    # ---- Draw series (plot others first, then Human last so it's on top) ----
    human_present = len(labels) > 0 and labels[0] == "Human"
    # others (indices 1..end) first
    start_idx = 1 if human_present else 0
    for mi in range(start_idx, n_models):
        row = list(values[mi]) + [values[mi][0]]
        ax.plot(angles, row, color=colors[mi], linewidth=1.8, label=labels[mi], zorder=3)
        ax.fill(angles, row, color=colors[mi], alpha=0.18, zorder=2)

    # Human last (on top)
    if human_present:
        hrow = list(values[0]) + [values[0][0]]
        ax.plot(angles, hrow, color=colors[0], linewidth=3.2, label=labels[0], zorder=10)
        ax.scatter(angles[:-1], values[0], color=colors[0], s=45, zorder=11)
        ax.fill(angles, hrow, color=colors[0], alpha=0.35, zorder=5)

    # ---- Radius & ticks ----
    max_val = np.nanmax(values)
    if not np.isfinite(max_val) or max_val <= 0:
        max_val = 1.0
    outer_radius = max_val * (1.0 + max(label_radius_pad, 0.06))
    ax.set_ylim(0, outer_radius)

    yticks = np.linspace(0, max_val, num=int(np.ceil(max_val)) + 1)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{y:.1f}" for y in yticks], fontsize=12)

    # ---- Spoke labels (outside) ----
    if pair_label_fontsize is None:
        pair_label_fontsize = max(10, int(28 - 0.7 * n_groups))
    ax.set_xticks([])

    r_label = max_val * (1.0 + label_radius_pad)
    effects = [pe.withStroke(linewidth=3, foreground="white")] if label_outline else None

    for angle, txt in zip(angles[:-1], display_pairs):
        a = (angle + np.pi) % (2 * np.pi) - np.pi
        ha = "left" if (-np.pi / 2 < a < np.pi / 2) else ("center" if abs(a) == np.pi / 2 else "right")
        ax.text(
            angle,
            r_label,
            txt,
            ha=ha,
            va="center",
            fontsize=pair_label_fontsize,
            path_effects=effects,
        )

    plt.tight_layout()

    # ---- Save figure ----
    if output_path:
        plt.savefig(output_path, bbox_inches="tight", dpi=300)

    # ---- Legend (Human first) ----
    leg_handles, leg_labels = ax.get_legend_handles_labels()
    # Build label->handle (use last occurrence) and then order by our 'labels' list
    handle_by_label = {lab: h for h, lab in zip(leg_handles, leg_labels)}
    ordered_handles = [handle_by_label[lab] for lab in labels if lab in handle_by_label]

    legend_fig, legend_ax = plt.subplots(figsize=(14, 1))
    legend_ax.axis("off")
    legend_ax.legend(
        ordered_handles,
        labels,  # already Human-first
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
                f"values_results/{model_label}/experiments_results.json",
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

    os.makedirs(f"vsm_images/{scenario}/", exist_ok=True)

    output_path = (
        f"vsm_images/{scenario}/{specific_name}.pdf"
        if specific_name is not None
        else f"vsm_images/{scenario}/{attribute}.pdf"
    )
    csv_path = (
        f"vsm_images/{scenario}/{specific_name}.csv"
        if specific_name is not None
        else f"vsm_images/{scenario}/{attribute}.csv"
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
                f"values_results/{model_label}/experiments_results.json",
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

    os.makedirs(f"vsm_images/{scenario}/", exist_ok=True)

    output_path = (
        f"vsm_images/{scenario}/{specific_name}_heatmap.pdf"
        if specific_name is not None
        else f"vsm_images/{scenario}/{attribute}_heatmap.pdf"
    )
    # csv_path = (
    #     f"vsm_images/{scenario}/{specific_name}.csv"
    #     if specific_name is not None
    #     else f"vsm_images/{scenario}/{attribute}.csv"
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


def display_model_consistency_comparison(model_list):
    """For generating heatmap figure

    Args:
        model_list (_type_): _description_
    """
    ratio_dict = {}
    for model_label in model_list:
        try:
            with open(
                f"values_results/{model_label}/experiments_results_2.json",
                "r",
                encoding="utf-8",
            ) as jl_file:
                experiments_results = json.load(jl_file)["cross_datasets_results"]
                ratio_dict[model_label] = experiments_results["per_user"]["ratios"]
        except Exception as e:
            print(model_label)
            print(str(e))

    model_comparison_stats = compare_models_stats(ratio_dict=ratio_dict)

    with open(
        "vsm_images/consistency_comparison/consistency_comparison_results_2.json",
        "w",
        encoding="utf-8",
    ) as c_f:
        json.dump(model_comparison_stats, c_f, indent=2)


edu_group_to_abbrev = {
    "High School": "HS",
    "Bachelor's Degree": "BD",
    "Master's Degree": "MD",
    "PhD": "PHD",
}

dev_group_to_abbrev = {"Developing": "DEV", "Developed": "DEVD", "Third World": "TW"}


display_model_consistency_comparison(
    [
        "Llama3.1-8B-Instruct",
        "Llama3.1-70B-Instruct",
        "DeepSeek-V3",
        "Qwen2.5-7B-Instruct",
        "Qwen2.5-72B-Instruct",
        "QwQ-32B",
    ],
)

# BA User

# display_comparison(
#     [
#         "Llama3.1-8B-Instruct",
#         "Llama3.1-70B-Instruct",
#         "DeepSeek-V3",
#         "Qwen2.5-7B-Instruct",
#         "Qwen2.5-72B-Instruct",
#         "QwQ-32B",
#     ],
#     cmap="tab10",
#     attribute="Age",
#     scenario="ba_user",
#     specific_name="ba_user_age_radar",
#     # pair_label_fontsize=14,
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
#     cmap="tab10",
#     attribute="Education Level",
#     scenario="ba_user",
#     specific_name="ba_user_education_radar",
#     # pair_label_fontsize=14,
#     group_to_abbrev=edu_group_to_abbrev,
# )

display_comparison(
    [
        "Llama3.1-8B-Instruct",
        "Llama3.1-70B-Instruct",
        "DeepSeek-V3",
        "Qwen2.5-7B-Instruct",
        "Qwen2.5-72B-Instruct",
        "QwQ-32B",
    ],
    cmap="tab10",
    attribute="Development",
    scenario="ba_user",
    specific_name="ba_user_development_radar",
    # pair_label_fontsize=14,
    group_to_abbrev=dev_group_to_abbrev,
)


# # Career Dialogue
# display_comparison(
#     [
#         "Llama3.1-8B-Instruct",
#         "Llama3.1-70B-Instruct",
#         "DeepSeek-V3",
#         "Qwen2.5-7B-Instruct",
#         "Qwen2.5-72B-Instruct",
#         "QwQ-32B",
#     ],
#     cmap="tab10",
#     attribute="Education Level",
#     scenario="ba_dialogue_career",
#     specific_name="ba_dialogue_education_radar",
#     # pair_label_fontsize=14,
#     group_to_abbrev=edu_group_to_abbrev,
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
#     cmap="tab10",
#     attribute="Age",
#     scenario="ba_dialogue_career",
#     specific_name="ba_dialogue_career_age_radar",
#     # pair_label_fontsize=14,
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
#     cmap="tab10",
#     attribute="Development",
#     scenario="ba_dialogue_career",
#     specific_name="ba_dialogue_career_development_radar",
#     # pair_label_fontsize=14,
#     group_to_abbrev=dev_group_to_abbrev,
# )
