from typing import List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

AnswerVec = Union[Mapping[str, int], Sequence[int]]


def _get_answer(q: str, v: AnswerVec) -> Optional[int]:
    """Fetch the picked answer for question q from vector v (dict or list aligned to questions)."""
    if isinstance(v, Mapping):
        return v.get(q, None)
    # If it's a sequence, we assume the caller iterates in the same order of `questions`
    # and passes the sequence zipped with `questions`. We'll handle that in the main fn.
    raise TypeError("If v_a/v_b are sequences, call emd_between_vectors with `vector_mode='sequence'`.")


def _normalize(dist: float, lo: int, hi: int, normalize_range: bool) -> float:
    if normalize_range and hi > lo:
        return dist / (hi - lo)
    return float(dist)


def _safe_isnan(x):
    try:
        return np.isnan(x)
    except Exception:
        return False


def emd_between_vectors(
    questions: Sequence[str],
    v_a: AnswerVec,
    v_b: AnswerVec,
    question_metadata: Mapping[str, Mapping[str, Union[int, str]]],
    *,
    weights: Optional[Sequence[float]] = None,
    normalize_range: bool = True,
    vector_mode: str = "mapping",  # "mapping" for dicts {qid: answer}, "sequence" for lists aligned to `questions`
) -> Tuple[float, np.ndarray]:
    """
    Compute the EMD (1-Wasserstein) between two single-response vectors across multiple ordinal questions.

    Per question: distance = |a_i - b_i|, optionally normalized by (hi - lo).
    Aggregate: mean or weighted mean across questions with valid answers.

    Parameters
    ----------
    questions : list of question ids
    v_a, v_b : either dict-like {question_id: int} or sequences aligned to `questions` (set vector_mode accordingly)
    question_metadata : mapping with keys:
        - answer_scale_min (int), answer_scale_max (int)
        - (others ignored here)
    weights : optional sequence of nonnegative weights aligned to `questions`
    normalize_range : if True, divide per-question distance by (hi - lo) to get [0,1]
    vector_mode : "mapping" (default) or "sequence"
        - "mapping": v_a and v_b are dict-like and looked up by question id
        - "sequence": v_a and v_b are sequences with same length as `questions`

    Returns
    -------
    agg_emd : float
        (Weighted) mean of per-question distances over questions with valid answers.
    per_q : np.ndarray
        Per-question distances (np.nan where an answer is missing/invalid).
    """
    q_list = list(questions)
    n = len(q_list)

    # Prepare answers per question
    a_vals: List[Optional[int]] = []
    b_vals: List[Optional[int]] = []

    if vector_mode == "mapping":
        for q in q_list:
            a_vals.append(_get_answer(q, v_a))
            b_vals.append(_get_answer(q, v_b))
    elif vector_mode == "sequence":
        if not (isinstance(v_a, Sequence) and isinstance(v_b, Sequence)):
            raise TypeError("For vector_mode='sequence', v_a and v_b must be sequences.")
        if len(v_a) != n or len(v_b) != n:
            raise ValueError("When vector_mode='sequence', v_a and v_b must match len(questions).")
        a_vals = list(v_a)
        b_vals = list(v_b)
    else:
        raise ValueError("vector_mode must be 'mapping' or 'sequence'.")

    per_q = np.full(n, np.nan, dtype=float)

    for i, q in enumerate(q_list):
        q_info = question_metadata[q]
        lo, hi = int(q_info["answer_scale_min"]), int(q_info["answer_scale_max"])

        a_i, b_i = a_vals[i], b_vals[i]
        # Validate presence and range
        if a_i is None or b_i is None:
            continue
        if _safe_isnan(a_i) or _safe_isnan(b_i):
            continue
        if not (lo <= int(a_i) <= hi) or not (lo <= int(b_i) <= hi):
            # Out-of-range answers are skipped; you can choose to clamp or raise instead
            continue

        dist = abs(int(a_i) - int(b_i))
        per_q[i] = _normalize(dist, lo, hi, normalize_range)

    # Aggregate over valid entries
    valid = ~np.isnan(per_q)
    if not np.any(valid):
        return 0.0, per_q  # no comparable questions; define aggregate as 0.0

    if weights is None:
        agg = float(np.nanmean(per_q))
        return agg, per_q

    w = np.asarray(weights, dtype=float)
    if w.shape[0] != n:
        raise ValueError("weights must have the same length as `questions`.")
    # Use only weights where per_q is valid; renormalize
    w_valid = w[valid]
    if np.all(w_valid == 0):
        # degenerate weights; fall back to unweighted mean over valid
        agg = float(np.nanmean(per_q))
        return agg, per_q
    w_valid = w_valid / w_valid.sum()
    agg = float(np.nansum(per_q[valid] * w_valid))
    return agg, per_q
