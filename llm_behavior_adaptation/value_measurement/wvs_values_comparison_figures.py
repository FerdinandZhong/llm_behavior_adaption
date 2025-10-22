import json
import os
import re
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import get_cmap
from matplotlib.patches import Rectangle

mpl.rcParams["font.family"] = "arial"


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
    avg_divergence = [
        item["compared_details"]["average_user_divergence"] for item in data
    ]
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
    plt.axhline(
        y=baseline, color="red", linestyle="--", label=f"Baseline ({baseline:.3f})"
    )

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
    label_pad=-5,
):
    """
    Visualize divergence comparisons as relative ratios over baseline across multiple models using a radar chart,
    and optionally export the data to CSV.

    Parameters:
        datasets (list of lists): Multiple divergence datasets
        baselines (list): Baseline values for each model
        labels (list): Model names
        attribute (str): The attribute name (used in plot title)
        figsize (tuple): Figure dimensions
        output_path (str): Optional path to save the image
        csv_path (str): Optional path to save the underlying data as CSV
        cmap (str): Colormap name for plotting
    """
    # Validation
    if not (len(datasets) == len(baselines) == len(labels)):
        raise ValueError("All input lists must have equal length")

    # Extract baselines
    baselines = [b["overall_baseline"] for b in baselines]
    n_models = len(datasets)

    # Collect and sort groups
    all_groups = set()
    for data in datasets:
        for item in data:
            group_name = item["compared_groups"].replace("--", " vs ")
            if "unknown" not in group_name.lower():
                all_groups.add(group_name)
    sorted_groups = sorted(all_groups)
    n_groups = len(sorted_groups)

    # Prepare values matrix
    values = np.zeros((n_models, n_groups))
    for mi, data in enumerate(datasets):
        base = baselines[mi]
        for item in data:
            group = item["compared_groups"].replace("--", " vs ")
            if "unknown" not in group.lower():
                idx = sorted_groups.index(group)
                div = item["compared_details"]["average_divergence"]
                values[mi, idx] = div / base

    # Export to CSV if requested
    if csv_path:
        df = pd.DataFrame(
            {
                "Group": sorted_groups,
                **{labels[mi]: values[mi, :] for mi in range(n_models)},
            }
        )
        df.to_csv(csv_path, index=False)

    # Radar chart setup
    angles = np.linspace(0, 2 * np.pi, n_groups, endpoint=False).tolist()
    angles += angles[:1]  # close the loop

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    cmap = get_cmap(cmap)
    colors = [cmap(i % 20) for i in range(n_models)]

    # Plot each model
    for mi in range(n_models):
        data_row = list(values[mi]) + [values[mi][0]]
        ax.plot(angles, data_row, color=colors[mi], linewidth=2, label=labels[mi])
        ax.fill(angles, data_row, color=colors[mi], alpha=0.25)

    # Add group labels with rotation to reduce overlap
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(
        sorted_groups,
        fontsize=23,
        ha="center",
        fontdict={"family": "arial"},
    )
    for label in ax.get_xticklabels():
        label.set_y(label.get_position()[1] - (label_pad / 100))  # Apply custom padding

    # Set radius labels
    max_val = np.nanmax(values)
    yticks = np.linspace(0, max_val, num=int(max_val) + 1)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{y:.1f}" for y in yticks], fontsize=15)
    ax.set_ylim(0, max_val)

    plt.tight_layout()

    # Save to file if output path is specified
    if output_path:
        plt.savefig(output_path, bbox_inches="tight", dpi=300)

    legend_fig, legend_ax = plt.subplots(figsize=(12, 1))
    legend_ax.axis("off")
    legend = legend_ax.legend(
        *ax.get_legend_handles_labels(),
        loc="center",
        fontsize=30,
        frameon=False,
        ncol=min(6, n_models),
    )
    if output_path:
        legend_fig.savefig(
            output_path.replace(".pdf", "_legend.pdf"), bbox_inches="tight", dpi=300
        )


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
    order_pairs: Optional[
        Callable[[List[str]], List[str]]
    ] = None,  # full custom column order
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
        mat = mat.loc[
            [emphasize_label] + [r for r in mat.index if r != emphasize_label], :
        ]

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

    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    # cbar.set_label("Ratio over baseline", rotation=90)

    if emphasize_label in mat.index:
        r = list(mat.index).index(emphasize_label)
        plt.gca().add_patch(
            Rectangle((-0.5, r - 0.5), mat.shape[1], 1, fill=False, lw=2)
        )

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
                    experiments_results = json.load(jl_file)[f"{scenario}_results"][
                        attribute
                    ]
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
                    experiments_results = json.load(jl_file)[f"{scenario}_results"][
                        attribute
                    ]
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


# display_comparison(
#     [
#         "Human",
#         "Llama-3.1-8B-Instruct",
#         "Llama-3.1-70B-Instruct",
#         "DeepSeek-V3",
#         "Qwen2.5-7B-Instruct",
#         "Qwen2.5-72B-Instruct",
#         # "QwQ-32B",
#     ],
#     cmap="tab10",
#     attribute="highest_level_of_education",
#     scenario="ba_user",
#     # extra_rules=["<30", ">60"],
#     specific_name="ba_user_education_radar",
# )
display_comparison_heatmap(
    [
        "Human",
        "Llama-3.1-8B-Instruct",
        "Llama-3.1-70B-Instruct",
        "DeepSeek-V3",
        "Qwen2.5-7B-Instruct",
        "Qwen2.5-72B-Instruct",
        # "QwQ-32B",
    ],
    cmap="viridis",
    attribute="age",
    scenario="ba_user",
    # extra_rules=["<30", ">60"],
    specific_name="ba_user_age",
    defined_order=["<30", "30-40", "40-50", "50-60", ">50"],
)


display_comparison_heatmap(
    [
        "Human",
        "Llama-3.1-8B-Instruct",
        "Llama-3.1-70B-Instruct",
        "DeepSeek-V3",
        "Qwen2.5-7B-Instruct",
        "Qwen2.5-72B-Instruct",
        # "QwQ-32B",
    ],
    cmap="viridis",
    attribute="highest_level_of_education",
    scenario="ba_user",
    # extra_rules=["<30", ">60"],
    specific_name="ba_user_education",
    defined_order=[
        "Basic education",
        "High school & equivalent",
        "Short-cycle tertiary",
        "Bachelor",
        "Master’s & Doctoral",
    ],
)

display_comparison_heatmap(
    [
        "Human",
        "Llama-3.1-8B-Instruct",
        "Llama-3.1-70B-Instruct",
        "DeepSeek-V3",
        "Qwen2.5-7B-Instruct",
        "Qwen2.5-72B-Instruct",
        # "QwQ-32B",
    ],
    cmap="viridis",
    attribute="socioeconomic_status",
    scenario="ba_user",
    # extra_rules=["<30", ">60"],
    specific_name="ba_user_socioeconomic_status",
    defined_order=[
        "Lower class",
        "Working class",
        "Lower middle class",
        "Upper middle class",
        "Upper class",
    ],
)

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
