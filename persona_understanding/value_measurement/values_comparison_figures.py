import matplotlib.pyplot as plt
import numpy as np
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
