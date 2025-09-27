import numpy as np
from scipy.stats import wasserstein_distance

# Assuming you already have question_metadata like:
# question_metadata[q] = {"answer_scale_min": a_min, "answer_scale_max": a_max}


def get_distribution(raw_answers, question, question_metadata):
    """Return probability over integer options for this question."""
    q_info = question_metadata[question]
    lo, hi = q_info["answer_scale_min"], q_info["answer_scale_max"]
    # edges centered on integers: [lo-0.5, lo+0.5, ..., hi+0.5]
    edges = np.arange(lo - 0.5, hi + 1.5, 1.0)
    counts, _ = np.histogram(raw_answers, bins=edges)
    total = counts.sum()
    return counts / total if total > 0 else np.zeros_like(counts, dtype=float)


def wasserstein_q(
    a_answers, b_answers, question, question_metadata, normalize_range=True
):
    """W1 between two groups on a single question."""
    q_info = question_metadata[question]
    lo, hi = q_info["answer_scale_min"], q_info["answer_scale_max"]
    support = np.arange(lo, hi + 1)  # integer categories as positions
    p = get_distribution(
        a_answers, question, question_metadata
    )  # probs align with support
    q = get_distribution(b_answers, question, question_metadata)

    w1 = wasserstein_distance(support, support, u_weights=p, v_weights=q)
    if normalize_range and hi > lo:
        w1 = w1 / (hi - lo)  # put into [0,1] range for comparability across questions
    return w1


def aggregate_wasserstein(
    questions, a_answers_by_q, b_answers_by_q, question_metadata, weights=None
):
    """Compute per-question W1 then aggregate (mean or weighted mean)."""
    w1s = np.array(
        [
            wasserstein_q(
                a_answers_by_q[q],
                b_answers_by_q[q],
                q,
                question_metadata,
                normalize_range=True,
            )
            for q in questions
        ]
    )
    if weights is None:
        return w1s.mean(), w1s
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    return (w1s * weights).sum(), w1s
