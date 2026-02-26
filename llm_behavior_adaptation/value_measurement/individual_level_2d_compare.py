import argparse
import json
import os
from typing import Dict, Iterable, List, Tuple

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from llm_behavior_adaptation.value_measurement.wvs_values_comparison import (  # noqa: E402
    ATTRIBUTES,
    DATASET_DIR,
    ValuesComparison,
    load_jsonl_file,
)

STYLE_RC_PARAMS = {
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
}

PALETTE = {
    "human": "#4C78A8",
    "ba_user": "#F58518",
    "ba_dialogue_career": "#54A24B",
    "ba_dialogue_investment": "#B279A2",
}


def _process_model_outputs(original_answers_list: List[Dict]) -> Dict[str, Dict[str, Dict]]:
    processed_answers_dict: Dict[str, Dict[str, Dict]] = {}
    for answer_details in original_answers_list:
        per_user_answers: Dict[str, Dict] = {}
        for user_id, answers in answer_details.items():
            for cat_answers in list(answers.values()):
                for answer in cat_answers:
                    per_user_answers.update(answer)
            if user_id in processed_answers_dict:
                print(user_id)
            processed_answers_dict[user_id] = per_user_answers
    return processed_answers_dict


def _vectorize_answers(
    answer_map: Dict[str, Dict],
    question_ids: Iterable[str],
) -> Tuple[List[str], np.ndarray]:
    qids = list(question_ids)
    ids = sorted(answer_map.keys())
    mat = np.full((len(ids), len(qids)), np.nan, dtype=float)
    for i, uid in enumerate(ids):
        per_q = answer_map.get(uid, {})
        for j, qid in enumerate(qids):
            val = per_q.get(qid)
            if isinstance(val, dict):
                val = val.get("option_id")
            if val is None:
                continue
            try:
                mat[i, j] = float(val)
            except Exception:
                continue
    return ids, mat


def _pca_2d(x: np.ndarray) -> np.ndarray:
    x = x.astype(float)
    # z-score with safe std
    col_mean = np.nanmean(x, axis=0)
    col_std = np.nanstd(x, axis=0)
    col_std[col_std == 0] = 1.0
    x = (x - col_mean) / col_std
    # fill remaining nans with 0 (mean after normalization)
    x = np.nan_to_num(x, nan=0.0)
    x = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    return x @ vt[:2].T


def _tsne_2d(x: np.ndarray, perplexity: float, random_state: int) -> np.ndarray:
    try:
        from sklearn.manifold import TSNE
    except Exception as exc:
        raise RuntimeError("scikit-learn is required for t-SNE. Install `scikit-learn`.") from exc

    x = x.astype(float)
    col_mean = np.nanmean(x, axis=0)
    x = np.nan_to_num(x - col_mean, nan=0.0)

    n_samples = x.shape[0]
    max_perp = max(2.0, (n_samples - 1) / 3.0)
    perp = min(perplexity, max_perp)

    tsne = TSNE(
        n_components=2,
        init="pca",
        perplexity=perp,
        random_state=random_state,
        learning_rate="auto",
    )
    return tsne.fit_transform(x)


def _embed_2d(x: np.ndarray, method: str, perplexity: float, random_state: int) -> np.ndarray:
    method = method.lower()
    if method == "pca":
        return _pca_2d(x)
    if method == "tsne":
        return _tsne_2d(x, perplexity=perplexity, random_state=random_state)
    raise ValueError(f"Unknown embedding method: {method}")


def _save_figure(fig: plt.Figure, out_path: str) -> None:
    root, ext = os.path.splitext(out_path)
    if ext.lower() != ".png":
        root = out_path
    fig.savefig(f"{root}.png", dpi=220)
    fig.savefig(f"{root}.pdf")


def _plot_side_by_side(
    attr: str,
    human_xy: np.ndarray,
    human_attr: List[str],
    model_xy: np.ndarray,
    model_attr: List[str],
    out_path: str,
    title_suffix: str,
    x_label: str,
    y_label: str,
) -> None:
    categories = sorted({a for a in human_attr if a is not None})
    color_map = {cat: plt.cm.tab20(i % 20) for i, cat in enumerate(categories)}

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), sharex=True, sharey=True)
    for ax, xy, labels, title in [
        (axes[0], human_xy, human_attr, "Human (user values)"),
        (axes[1], model_xy, model_attr, "Model outputs"),
    ]:
        for cat in categories:
            idx = [i for i, v in enumerate(labels) if v == cat]
            if not idx:
                continue
            pts = xy[idx]
            ax.scatter(pts[:, 0], pts[:, 1], s=12, alpha=0.75, color=color_map[cat], label=cat)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=3,
            fontsize=9,
            frameon=False,
            prop={"weight": "bold"},
        )
    fig.suptitle(f"{attr}: human vs model embeddings {title_suffix}", fontsize=12)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    _save_figure(fig, out_path)
    plt.close(fig)


def _plot_overlay(
    human_xy: np.ndarray,
    model_xy: np.ndarray,
    out_path: str,
    title: str,
    model_color: str,
    x_label: str,
    y_label: str,
) -> None:
    plt.figure(figsize=(6.8, 6.2))
    plt.scatter(
        human_xy[:, 0],
        human_xy[:, 1],
        s=14,
        alpha=0.7,
        color=PALETTE["human"],
        label="Human",
    )
    plt.scatter(
        model_xy[:, 0],
        model_xy[:, 1],
        s=10,
        alpha=0.5,
        color=model_color,
        label="Model",
    )
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(loc="best", frameon=False, prop={"weight": "bold"})
    plt.tight_layout()
    fig = plt.gcf()
    _save_figure(fig, out_path)
    plt.close(fig)


def _plot_overlay_all(
    human_xy: np.ndarray,
    ba_user_xy: np.ndarray,
    career_xy: np.ndarray,
    investment_xy: np.ndarray,
    out_path: str,
    title: str,
    x_label: str,
    y_label: str,
) -> None:
    plt.figure(figsize=(7.2, 6.4))
    plt.scatter(human_xy[:, 0], human_xy[:, 1], s=14, alpha=0.7, color=PALETTE["human"], label="Human")
    plt.scatter(ba_user_xy[:, 0], ba_user_xy[:, 1], s=10, alpha=0.45, color=PALETTE["ba_user"], label="BA_user")
    plt.scatter(
        career_xy[:, 0],
        career_xy[:, 1],
        s=10,
        alpha=0.45,
        color=PALETTE["ba_dialogue_career"],
        label="BA_dialogue_career",
    )
    plt.scatter(
        investment_xy[:, 0],
        investment_xy[:, 1],
        s=10,
        alpha=0.45,
        color=PALETTE["ba_dialogue_investment"],
        label="BA_dialogue_investment",
    )
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(frameon=False, fontsize=9, prop={"weight": "bold"})
    plt.tight_layout()
    fig = plt.gcf()
    _save_figure(fig, out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Individual-level 2D embedding comparison")
    parser.add_argument(
        "--user-profile-dataset",
        type=str,
        default=f"{DATASET_DIR}/sampled_demographic_features.csv",
    )
    parser.add_argument(
        "--user-value-dataset",
        type=str,
        default=f"{DATASET_DIR}/sampled_values_df.csv",
    )
    parser.add_argument("--ba-user-results", type=str, required=True)
    parser.add_argument("--ba-dialogue-career-results", type=str, required=True)
    parser.add_argument("--ba-dialogue-investment-results", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--method",
        type=str,
        choices=["pca", "tsne"],
        default="pca",
        help="2D embedding method for individual-level scatter plots.",
    )
    parser.add_argument(
        "--tsne-perplexity",
        type=float,
        default=30.0,
        help="t-SNE perplexity (will be capped based on sample size).",
    )
    parser.add_argument(
        "--tsne-random-state",
        type=int,
        default=42,
        help="Random state for t-SNE.",
    )
    parser.add_argument(
        "--overlay-all",
        action="store_true",
        default=False,
        help="If set, generate one overlay that includes human and all model outputs.",
    )
    parser.add_argument(
        "--skip-per-model",
        action="store_true",
        default=False,
        help="If set, skip per-model overlays and attribute-colored plots.",
    )
    args = parser.parse_args()

    plt.rcParams.update(STYLE_RC_PARAMS)

    user_profile = pd.read_csv(args.user_profile_dataset)
    user_value = pd.read_csv(args.user_value_dataset)

    with open(f"{DATASET_DIR}/picked_questions.json", "r", encoding="utf-8") as f:
        picked_questions = json.load(f)

    vc = ValuesComparison(
        user_profile_dataset=user_profile,
        user_value_dataset=user_value,
        ba_user_results=_process_model_outputs(load_jsonl_file(args.ba_user_results)),
        ba_dialogue_career_results=_process_model_outputs(load_jsonl_file(args.ba_dialogue_career_results)),
        ba_dialogue_investment_results=_process_model_outputs(load_jsonl_file(args.ba_dialogue_investment_results)),
        picked_questions=picked_questions,
        results_output_path="",
        verbose=0,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    # Prepare human vectors
    question_ids = list(vc.all_questions.keys())
    human_df = vc.user_value_dataset.copy()
    human_df["D_INTERVIEW"] = human_df["D_INTERVIEW"].astype(str)
    human_df = human_df.set_index("D_INTERVIEW")
    human_ids = human_df.index.tolist()
    human_mat = human_df[question_ids].to_numpy(dtype=float)

    model_maps = {
        "ba_user": vc.ba_user_results,
        "ba_dialogue_career": vc.ba_dialogue_career_results,
        "ba_dialogue_investment": vc.ba_dialogue_investment_results,
    }

    profile_df = vc.user_profile_dataset.copy()
    profile_df["D_INTERVIEW"] = profile_df["D_INTERVIEW"].astype(str)
    profile_df = profile_df.set_index("D_INTERVIEW")

    if args.method == "pca":
        x_label, y_label = "PC1", "PC2"
    else:
        x_label, y_label = "t-SNE 1", "t-SNE 2"

    if args.overlay_all:
        ba_user_ids, ba_user_mat = _vectorize_answers(model_maps["ba_user"], question_ids)
        career_ids, career_mat = _vectorize_answers(model_maps["ba_dialogue_career"], question_ids)
        investment_ids, investment_mat = _vectorize_answers(model_maps["ba_dialogue_investment"], question_ids)
        combined_all = np.vstack([human_mat, ba_user_mat, career_mat, investment_mat])
        combined_xy = _embed_2d(
            combined_all,
            method=args.method,
            perplexity=args.tsne_perplexity,
            random_state=args.tsne_random_state,
        )
        n_h = len(human_mat)
        n_u = len(ba_user_mat)
        n_c = len(career_mat)
        human_xy = combined_xy[:n_h]
        user_xy = combined_xy[n_h : n_h + n_u]
        career_xy = combined_xy[n_h + n_u : n_h + n_u + n_c]
        invest_xy = combined_xy[n_h + n_u + n_c :]

        overlay_all_path = os.path.join(args.output_dir, f"overlay_all_models_{args.method}.png")
        _plot_overlay_all(
            human_xy=human_xy,
            ba_user_xy=user_xy,
            career_xy=career_xy,
            investment_xy=invest_xy,
            out_path=overlay_all_path,
            title="Human vs Model Outputs",
            x_label=x_label,
            y_label=y_label,
        )

    if not args.skip_per_model:
        for model_key, model_map in model_maps.items():
            model_ids, model_mat = _vectorize_answers(model_map, question_ids)
            # Align by union to fit PCA
            combined = np.vstack([human_mat, model_mat])
            combined_xy = _embed_2d(
                combined,
                method=args.method,
                perplexity=args.tsne_perplexity,
                random_state=args.tsne_random_state,
            )
            human_xy = combined_xy[: len(human_mat)]
            model_xy = combined_xy[len(human_mat) :]

            # Head-to-head overlay plot (no attributes)
            overlay_path = os.path.join(args.output_dir, f"{model_key}_overlay.png")
            _plot_overlay(
                human_xy=human_xy,
                model_xy=model_xy,
                out_path=overlay_path,
                title="Human vs Model Outputs",
                model_color=PALETTE.get(model_key, PALETTE["ba_user"]),
                x_label=x_label,
                y_label=y_label,
            )

            for attr in ATTRIBUTES:
                human_attr = (
                    profile_df.loc[human_ids, attr].astype(str).replace({"nan": None}).tolist()
                    if attr in profile_df.columns
                    else [None] * len(human_ids)
                )
                model_attr = (
                    profile_df.loc[model_ids, attr].astype(str).replace({"nan": None}).tolist()
                    if attr in profile_df.columns
                    else [None] * len(model_ids)
                )

                out_path = os.path.join(args.output_dir, f"{model_key}_{attr}.png")
                _plot_side_by_side(
                    attr=attr,
                    human_xy=human_xy,
                    human_attr=human_attr,
                    model_xy=model_xy,
                    model_attr=model_attr,
                    out_path=out_path,
                    title_suffix=model_key,
                    x_label=x_label,
                    y_label=y_label,
                )


if __name__ == "__main__":
    main()
