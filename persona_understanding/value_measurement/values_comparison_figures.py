import json
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import get_cmap
from sklearn.manifold import TSNE


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


def plot_divergence_comparison(
    datasets,
    baselines,
    labels,
    attribute: str,
    figsize=(16, 12),
    output_path=None,
    group_spacing=1.5,
    cmap="tab20",
):
    """
    Visualize divergence comparisons as relative ratios over baseline across multiple models.

    Parameters:
        datasets (list of lists): Multiple divergence datasets
        baselines (list): Baseline values for each model
        labels (list): Model names
        figsize (tuple): Figure dimensions
        output_path (str): Optional path to save the image
        group_spacing (float): Vertical space between comparison groups (default: 1.5)
    """
    # Validation
    if not len(datasets) == len(baselines) == len(labels):
        raise ValueError("All input lists must have equal length")

    # Extract baselines and prepare data
    baselines = [b["overall_baseline"] for b in baselines]
    n_models = len(datasets)

    # Collect and sort groups
    all_groups = set()
    for data in datasets:
        for item in data:
            group_name = item["compared_groups"].replace("--", " vs ")
            if "unknown" not in group_name.lower():  # Ignore groups with "unknown"
                all_groups.add(group_name)

    # Enhanced sorting: < groups first, then numeric groups, then others
    def sort_key(group):
        if "<" in group:
            return (0, group)
        return (1, group)

    sorted_groups = sorted(all_groups, key=sort_key)
    group_indices = {g: i for i, g in enumerate(sorted_groups)}

    # Prepare values matrix
    n_groups = len(sorted_groups)
    print(n_groups)
    values = np.full((n_models, len(sorted_groups)), np.nan)
    for mi, data in enumerate(datasets):
        base = baselines[mi]
        for item in data:
            group = item["compared_groups"].replace("--", " vs ")
            if "unknown" not in group.lower():  # Skip "unknown" groups
                div = item["compared_details"]["average_divergence"]
                values[mi, group_indices[group]] = div / base

    # Plot configuration
    plt.figure(figsize=figsize)
    ax = plt.gca()
    cmap = get_cmap(cmap)
    colors = [cmap(i % 20) for i in range(n_models)]

    # Dynamic spacing based on number of models
    group_positions = np.arange(n_groups) * group_spacing  # Main change here

    # Modified bar positioning
    total_width = min(1.2, 0.6 + 0.1 * len(datasets))
    bar_width = total_width / len(datasets)

    # Plot bars for each model
    for mi in range(len(datasets)):
        # Adjusted position calculation using group_spacing
        x_pos = group_positions - total_width / 2 + bar_width / 2 + mi * bar_width

        bars = ax.barh(
            x_pos,
            values[mi],
            height=bar_width,  # Height relative to group spacing
            color=colors[mi],
            alpha=0.85,
            label=f"{labels[mi]}",
        )

        # Add value labels with dynamic positioning
        # for idx, val in enumerate(values[mi]):
        #     if not np.isnan(val):
        #         x_pos = val + (0.02 if val >= 0 else -0.02)
        #         ha = 'left' if val >= 0 else 'right'
        #         ax.text(x_pos, bars[idx].get_y() + bar_width/2,
        #                 f'{val:.2f}',
        #                 va='center', ha=ha,
        #                 fontsize=12, color=colors[mi])

    # Styling improvements
    ax.set_yticks(group_positions)
    ax.set_yticklabels(sorted_groups, fontsize=22, rotation=30)
    ax.invert_yaxis()
    ax.xaxis.grid(True, linestyle="--", alpha=0.4)

    # Baseline markers
    for mi in range(n_models):
        ax.axvline(1.0, color=colors[mi], linestyle=":", alpha=0.7, lw=2)

    # Labels and titles
    ax.set_xlabel("Relative Ratio of Divergence over Baseline", fontsize=22)
    ax.set_ylabel("Group Comparisons", fontsize=22)
    ax.set_title(
        f"Divergence Comparison ({attribute.capitalize()})", fontsize=24, pad=20
    )

    # Enhanced legend
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=min(4, n_models),
        fontsize=22,
        frameon=False,
    )

    # Final adjustments
    plt.xticks(fontsize=18)
    max_val = np.nanmax(np.abs(values)) * 1.1
    plt.xlim(-0.1 if np.any(values < 0) else 0, max_val)
    plt.tight_layout()

    # Save to file if output path is specified
    if output_path:
        plt.savefig(output_path, bbox_inches="tight", dpi=300)

    # plt.show()


def display_comparison(
    model_list,
    scenario: str = "BA_user",
    attribute: str = "age",
    group_spacing=1.25,
    cmap="tab20",
):
    datasets = []
    baselines = []
    for model_label in model_list:
        try:
            with open(
                f"../values_results/{model_label}/vsm/{scenario}/{attribute}.jsonl", "r"
            ) as jl_file:
                dataset = []
                for result in jl_file.readlines():
                    result = json.loads(result)
                    # if result["compared_groups"].startswith("<30"):
                    dataset.append(result)
                datasets.append(dataset)
            with open(
                f"../values_results/{model_label}/vsm/{scenario}/{attribute}_baseline.json",
                "r",
            ) as j_file:
                baselines.append(json.load(j_file))
        except Exception as e:
            print(model_label)
            print(str(e))

    os.makedirs(f"../images/{scenario}/", exist_ok=True)
    output_path = f"../images/{scenario}/{attribute}.png"
    plot_divergence_comparison(
        datasets=datasets,
        baselines=baselines,
        labels=model_list,
        attribute=attribute,
        output_path=output_path,
        group_spacing=group_spacing,
        cmap=cmap,
    )


display_comparison(
    [
        "Llama3.1-8B-Instruct",
        "Llama3.1-70B-Instruct",
        "DeepSeek-V3",
        "Qwen2.5-7B-Instruct",
        "Qwen2.5-72B-Instruct",
        "QwQ-32B",
    ],
    group_spacing=1.75,
    cmap="tab10",
    attribute="age",
    scenario="BA_dialogue",
)

# display_comparison(
#     [
#         "Llama3.1-8B-Instruct",
#         "Llama3.1-70B-Instruct",
#         "DeepSeek-V3",
#         "Qwen2.5-7B-Instruct",
#         "Qwen2.5-72B-Instruct",
#         "QwQ-32B"
#     ],
#     group_spacing=1.75,
#     cmap="tab10",
#     attribute="education",
#     scenario="BA_dialogue"
# )

# display_comparison(
#     [
#         "Llama3.1-8B-Instruct",
#         "Llama3.1-70B-Instruct",
#         "DeepSeek-V3",
#         "Qwen2.5-7B-Instruct",
#         "Qwen2.5-72B-Instruct",
#         "QwQ-32B"
#     ],
#     group_spacing=1.5,
#     cmap="tab10",
#     attribute="location",
#     scenario="BA_dialogue"
# )

# display_comparison(
#     [
#         "Llama3.1-8B-Instruct",
#         "Llama3.1-70B-Instruct",
#         "DeepSeek-V3",
#         "Qwen2.5-7B-Instruct",
#         "Qwen2.5-72B-Instruct",
#         "QwQ-32B"
#     ],
#     group_spacing=1.5,
#     cmap="tab10",
#     attribute="development_level",
#     scenario="BA_dialogue"
# )

# display_comparison(
#     [
#         "Llama3.1-8B-Instruct",
#         "Llama3.1-70B-Instruct",
#         "DeepSeek-V3",
#         "Qwen2.5-7B-Instruct",
#         "Qwen2.5-72B-Instruct",
#         "QwQ-32B"
#     ],
#     group_spacing=1.5,
#     cmap="tab10",
#     attribute="position_level",
#     scenario="BA_dialogue"
# )
