import numpy as np
from scipy.special import rel_entr


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
    if not np.allclose(p.sum(), 1) or not np.allclose(q.sum(), 1):
        raise ValueError("Both input distributions must sum to 1.")

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
    if not np.allclose(p.sum(), 1) or not np.allclose(q.sum(), 1):
        raise ValueError("Both input distributions must sum to 1.")

    # Compute the Hellinger distance
    return np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q))**2))
