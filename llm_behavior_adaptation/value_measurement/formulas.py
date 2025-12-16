import logging
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.special import rel_entr

logger = logging.getLogger(__name__)


def jensen_shannon_divergence(p, q, base=2):
    """
    Compute the Jensen-Shannon Divergence (JSD) between two probability distributions.

    Parameters:
        p (array-like): First probability distribution. Must sum to 1.
        q (array-like): Second probability distribution. Must sum to 1.
        base (float, optional): Logarithm base to use. Default is 2 for bits.

    Returns:
        float: Jensen-Shannon Divergence value.
    """
    p = np.asarray(p)
    q = np.asarray(q)

    # Ensure valid probability distributions
    if not np.allclose(p.sum(), 1):
        logger.warning(f"{p} is not sum to 1")
        return 1
    if not np.allclose(q.sum(), 1):
        logger.warning(f"{q} is not sum to 1")
        return 1

    # Compute the average distribution
    m = 0.5 * (p + q)

    # Compute KL divergences
    kl_pm = np.sum(rel_entr(p, m))
    kl_qm = np.sum(rel_entr(q, m))

    # Jensen-Shannon Divergence
    jsd = 0.5 * kl_pm + 0.5 * kl_qm

    # Convert to the desired log base
    if base != np.e:  # Convert if base is not the natural logarithm
        jsd /= np.log(base)

    return jsd


def hellinger_distance(p, q):
    """
    Compute the Hellinger Distance between two probability distributions.

    Parameters:
        p (array-like): First probability distribution. Must sum to 1.
        q (array-like): Second probability distribution. Must sum to 1.

    Returns:
        float: Hellinger Distance.
    """
    p = np.asarray(p)
    q = np.asarray(q)

    # Ensure valid probability distributions
    if not np.allclose(p.sum(), 1):
        logger.warning(f"{p} is not sum to 1")
        return 1
    if not np.allclose(q.sum(), 1):
        logger.warning(f"{q} is not sum to 1")
        return 1

    # Compute the Hellinger distance
    return np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2))


def softmax(z):
    """Compute the softmax of vector z."""
    e_z = np.exp(z - np.max(z))  # Numerical stability
    return e_z / e_z.sum()


def filter_rows(arr, rtol=1e-05, atol=1e-08):
    """filter the invalid distributions"""
    row_sums = arr.sum(axis=1)
    mask = np.isclose(row_sums, 1.0, rtol=rtol, atol=atol)
    filtered_arr = arr[mask]
    removed_count = arr.shape[0] - filtered_arr.shape[0]
    return filtered_arr, removed_count


def compute_js_centroid(distributions, maxiter=1000, tol=1e-6):
    """
    Compute the Jensen-Shannon centroid of a set of probability distributions.

    Parameters:
    - distributions: List of probability distributions (each as a numpy array).
    - maxiter: Maximum number of iterations for the optimization algorithm.
    - tol: Tolerance for convergence.

    Returns:
    - centroid: The computed Jensen-Shannon centroid as a numpy array.
    """
    valid_distributions, failed_count = filter_rows(distributions)

    # Normalize input distributions to ensure they are valid probabilities
    distributions = [np.array(p) / np.sum(p) for p in valid_distributions]

    # Compute initial centroid as the average of the input distributions
    initial_centroid = np.mean(valid_distributions, axis=0)

    # Avoid zeros in initial centroid for valid log transformation
    initial_centroid = np.clip(initial_centroid, 1e-8, None)
    initial_centroid = initial_centroid / initial_centroid.sum()

    # Convert initial centroid to logits (softmax parameters)
    z_init = np.log(initial_centroid)

    # Define the objective function to minimize (sum of JSDs)
    def objective(z):
        m = softmax(z)
        return np.sum([jensen_shannon_divergence(p, m) for p in valid_distributions])

    # Optimize using the L-BFGS-B algorithm
    result = minimize(objective, z_init, method="L-BFGS-B", options={"maxiter": maxiter, "ftol": tol})

    # Extract the optimized centroid
    centroid = softmax(result.x)
    return centroid, failed_count


def compute_js_centroid_and_avg(distributions, maxiter=1000, tol=1e-6):
    """
    Compute the Jensen-Shannon centroid of a set of probability distributions.

    Parameters:
    - distributions: List of probability distributions (each as a numpy array).
    - maxiter: Maximum number of iterations for the optimization algorithm.
    - tol: Tolerance for convergence.

    Returns:
    - centroid: The computed Jensen-Shannon centroid as a numpy array.
    """
    valid_distributions, failed_count = filter_rows(distributions)

    # Normalize input distributions to ensure they are valid probabilities
    distributions = [np.array(p) / np.sum(p) for p in valid_distributions]

    # Compute initial centroid as the average of the input distributions
    initial_centroid = np.mean(valid_distributions, axis=0)

    # Avoid zeros in initial centroid for valid log transformation
    initial_centroid = np.clip(initial_centroid, 1e-8, None)
    initial_centroid = initial_centroid / initial_centroid.sum()
    # print(f"initial centroid: {initial_centroid}")

    # Convert initial centroid to logits (softmax parameters)
    z_init = np.log(initial_centroid)
    n = len(distributions)  # Number of distributions

    # Define the objective function to minimize (sum of JSDs)
    def objective(z):
        m = softmax(z)
        return np.sum([jensen_shannon_divergence(p, m) for p in valid_distributions])

    # Optimize using the L-BFGS-B algorithm
    result = minimize(objective, z_init, method="L-BFGS-B", options={"maxiter": maxiter, "ftol": tol})

    # Extract the optimized centroid
    centroid = softmax(result.x)

    total_divergence = result.fun  # Sum of JSDs
    avg_divergence = total_divergence / n  # Average JSD

    return centroid, failed_count, avg_divergence


def compute_emd(vector_a, vector_b):
    # Convert to histograms
    bins = np.arange(1, 6)
    hist_a = np.histogram(vector_a, bins=bins)[0]
    hist_b = np.histogram(vector_b, bins=bins)[0]

    # Normalize to probabilities
    p = hist_a / np.sum(hist_a)
    q = hist_b / np.sum(hist_b)

    # Compute CDFs
    cdf_p = np.cumsum(p)
    cdf_q = np.cumsum(q)

    # Sum absolute differences of CDFs (excluding last category)
    emd = np.sum(np.abs(cdf_p[:-1] - cdf_q[:-1]))
    return emd


def _is_finite_scalar(x: Any) -> bool:
    """True if x is a finite scalar (not NaN/None/inf)."""
    try:
        return (x is not None) and np.isfinite(x)
    except Exception:
        return False


def _vector_has_nan_or_nonfinite(
    v: Mapping[str, Any],
    question_metadata: Mapping[str, Mapping[str, Any]],
) -> bool:
    """Return True if ANY value in v (for keys present in metadata) is NaN/None/inf."""
    for q in question_metadata.keys():
        if q in v and not _is_finite_scalar(v[q]):
            return True
    return False


def emd_distance(
    a: Mapping[str, Any],
    b: Mapping[str, Any],
    question_metadata: Mapping[str, Mapping[str, Any]],
    normalize: bool = True,
    *,
    skip_out_of_range: bool = True,
) -> float:
    """Compute normalized EMD (L1 on ordinal scales) between two answer vectors.
    Assumes caller filtered out NaN/None/inf vectors; still guards per-question.
    """
    d = 0.0
    for q, info in question_metadata.items():
        if q not in a or q not in b:
            continue

        lo = int(info["answer_scale_min"])
        hi = int(info["answer_scale_max"])

        av, bv = a[q], b[q]
        # per-question guard (should rarely trigger after prefilter)
        if not (_is_finite_scalar(av) and _is_finite_scalar(bv)):
            continue

        try:
            ai = int(float(av))
            bi = int(float(bv))
        except Exception as e:
            print(str(e))
            continue

        if skip_out_of_range:
            if not (lo <= ai <= hi and lo <= bi <= hi):
                continue
        else:
            ai = min(hi, max(lo, ai))
            bi = min(hi, max(lo, bi))

        diff = abs(ai - bi)
        d += (diff / (hi - lo)) if (normalize and hi > lo) else diff

    return float(d)


def emd_distance_vsm(
    a: List[int],
    b: List[int],
    normalize: bool = True,
    *,
    skip_out_of_range: bool = True,
) -> float:
    """Compute normalized EMD (L1 on ordinal scales) between two answer vectors.
    Assumes caller filtered out NaN/None/inf vectors; still guards per-question.
    """
    d = 0.0
    for av, bv in zip(a, b):
        lo = 1
        hi = 5

        # per-question guard (should rarely trigger after prefilter)
        if not (_is_finite_scalar(av) and _is_finite_scalar(bv)):
            continue

        try:
            ai = int(float(av))
            bi = int(float(bv))
        except Exception as e:
            print(str(e))
            continue

        if skip_out_of_range:
            if not (lo <= ai <= hi and lo <= bi <= hi):
                continue
        else:
            ai = min(hi, max(lo, ai))
            bi = min(hi, max(lo, bi))

        diff = abs(ai - bi)
        d += (diff / (hi - lo)) if (normalize and hi > lo) else diff

    return float(d)


def emd_medoid_skip_nan(
    vectors: Sequence[Mapping[str, Any]],
    question_metadata: Mapping[str, Mapping[str, Any]],
    normalize: bool = True,
    *,
    skip_out_of_range: bool = True,
) -> Tuple[Mapping[str, Any], float, int, List[int]]:
    """Find the medoid under normalized EMD, **ignoring any vector with NaN/None/inf**.

    Returns:
        medoid_vector: the most central (existing) vector among the CLEAN set
        minimal_total_distance: sum of distances from medoid to all other CLEAN vectors
        excluded_count: number of vectors skipped due to NaN/None/inf
        excluded_indices: indices (w.r.t. input `vectors`) that were excluded

    Notes:
        - If after filtering there are 0 clean vectors: raises ValueError.
        - If exactly 1 clean vector: returns it with distance 0.0.
        - Pairwise cost is O(m^2) for m = num of clean vectors.
    """
    print(len(question_metadata))
    if len(vectors) == 0:
        raise ValueError("emd_medoid_skip_nan: empty `vectors`.")

    # Identify and drop any vectors with NaN/None/inf on any metadata question
    excluded_indices: List[int] = []
    clean_vectors: List[Mapping[str, Any]] = []
    clean_to_orig_index: List[int] = []

    for idx, v in enumerate(vectors):
        if _vector_has_nan_or_nonfinite(v, question_metadata):
            excluded_indices.append(idx)
            continue
        clean_vectors.append(v)
        clean_to_orig_index.append(idx)

    excluded_count = len(excluded_indices)
    m = len(clean_vectors)

    if m == 0:
        raise ValueError("emd_medoid_skip_nan: all vectors contain NaN/None/inf.")
    if m == 1:
        return clean_vectors[0], 0.0, excluded_count, excluded_indices

    # Compute sum of distances to all others for each clean vector
    dsum = np.zeros(m, dtype=float)
    for i in range(m):
        vi = clean_vectors[i]
        for j in range(i + 1, m):
            vj = clean_vectors[j]
            d = emd_distance(
                vi,
                vj,
                question_metadata,
                normalize,
                skip_out_of_range=skip_out_of_range,
            )
            dsum[i] += d
            dsum[j] += d

    best_idx = int(np.argmin(dsum))
    medoid_vec = clean_vectors[best_idx]
    min_total = float(dsum[best_idx])

    return medoid_vec, min_total, excluded_count, excluded_indices


def componentwise_centroid(
    vectors: Sequence[Mapping[str, Any]],
    question_metadata: Mapping[str, Mapping[str, Any]],
) -> Dict[str, int]:
    """
    Compute the component-wise centroid for a cluster:
    - per question, take the median of valid answers;
    - when there are two middle values, take their average and round();
    - clamp the result to [answer_scale_min, answer_scale_max].

    Returns:
        Dict[qid, int]: centroid answer per question.
    """
    centroid: Dict[str, int] = {}

    for q, info in question_metadata.items():
        lo = int(info["answer_scale_min"])
        hi = int(info["answer_scale_max"])

        vals = []
        for v in vectors:
            if q not in v:
                continue
            x = v[q]
            # skip None/NaN/inf
            try:
                if x is None or not np.isfinite(x):
                    continue
            except Exception:
                # if it can't be checked (e.g., string), try to coerce; else skip
                try:
                    x = float(x)
                    if not np.isfinite(x):
                        continue
                except Exception:
                    continue

            # coerce to int safely (accept 2.0, "3", etc.)
            try:
                xi = int(x)
            except Exception:
                try:
                    xi = int(float(x))
                except Exception:
                    continue

            # clamp to valid range
            xi = min(hi, max(lo, xi))
            vals.append(xi)

        if not vals:
            continue

        vals = np.sort(np.asarray(vals, dtype=int))
        m = len(vals)
        if m % 2 == 1:
            med = int(vals[m // 2])
        else:
            # always use "round" for the tie
            lower = int(vals[m // 2 - 1])
            upper = int(vals[m // 2])
            med = int(round((lower + upper) / 2))
            med = min(hi, max(lo, med))

        centroid[q] = med

    return centroid


def componentwise_centroid_vsm(
    vectors: Sequence[List[int]],
) -> List[int]:
    """
    Compute the component-wise centroid for a cluster:
    - per question, take the median of valid answers;
    - when there are two middle values, take their average and round();
    - clamp the result to [answer_scale_min, answer_scale_max].

    Returns:
        Dict[qid, int]: centroid answer per question.
    """
    centroid: List[int] = []

    questions_length = len(vectors[0])

    lo = 1
    hi = 5
    for q_idx in range(questions_length):
        vals = [vector[q_idx] for vector in vectors]

        if not vals:
            continue

        vals = np.sort(np.asarray(vals, dtype=int))
        m = len(vals)
        if m % 2 == 1:
            med = int(vals[m // 2])
        else:
            # always use "round" for the tie
            lower = int(vals[m // 2 - 1])
            upper = int(vals[m // 2])
            med = int(round((lower + upper) / 2))
            med = min(hi, max(lo, med))

        centroid.append(med)

    return centroid
