import json
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import get_cmap
from sklearn.manifold import TSNE

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


def visualize_tsne_groups(groups, perplexity=30, learning_rate=200, random_state=42):
    """
    Visualize 4 groups of probability distributions using t-SNE.

    Parameters:
        groups (list of np.ndarray): A list of 4 arrays, where each array represents a group.
                                     Each array has shape [n_samples, 5], where n_samples is between 200 and 300.
        perplexity (float): t-SNE perplexity parameter.
        learning_rate (float): t-SNE learning rate.
        random_state (int): Random seed for reproducibility.

    Returns:
        None (displays a 2D scatter plot with groups highlighted).
    """
    # Combine all groups into a single array
    combined_data = np.vstack(groups)  # Shape: [total_samples, 5]

    # Create labels for each group
    labels = np.concatenate([np.full(len(group), i) for i, group in enumerate(groups)])

    # Apply t-SNE
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=learning_rate,
        random_state=random_state,
    )
    embedded_data = tsne.fit_transform(combined_data)

    # Plot the results
    plt.figure(figsize=(10, 8))
    colors = plt.cm.tab10.colors[:4]  # Use 4 distinct colors for the groups
    for i in range(4):
        # Scatter plot for each group
        group_indices = labels == i
        plt.scatter(
            embedded_data[group_indices, 0],
            embedded_data[group_indices, 1],
            color=colors[i],
            label=f"Group {i+1}",
            alpha=0.6,
        )

        # Plot group centroid
        group_center = np.mean(embedded_data[group_indices], axis=0)
        plt.scatter(
            group_center[0],
            group_center[1],
            color=colors[i],
            edgecolor="black",
            s=200,
            marker="X",
        )

    plt.title("t-SNE Visualization of 4 Groups")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.legend()
    plt.grid(alpha=0.3)
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
            with open(f"wvs_values_results/{model_label}/experiments_results.json", "r", encoding="utf-8") as jl_file:
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
    )


display_comparison(
    [
        "Human",
        "Llama-3.1-8B-Instruct",
        "Llama-3.1-70B-Instruct",
        "DeepSeek-V3",
        "Qwen2.5-7B-Instruct",
        "Qwen2.5-72B-Instruct",
        # "QwQ-32B",
    ],
    cmap="tab10",
    attribute="highest_level_of_education",
    scenario="ba_user",
    # extra_rules=["<30", ">60"],
    specific_name="ba_user_education_radar",
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
