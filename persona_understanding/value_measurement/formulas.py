import logging

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
    result = minimize(
        objective, z_init, method="L-BFGS-B", options={"maxiter": maxiter, "ftol": tol}
    )

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
    result = minimize(
        objective, z_init, method="L-BFGS-B", options={"maxiter": maxiter, "ftol": tol}
    )

    # Extract the optimized centroid
    centroid = softmax(result.x)

    total_divergence = result.fun  # Sum of JSDs
    avg_divergence = total_divergence / n  # Average JSD

    return centroid, failed_count, avg_divergence
