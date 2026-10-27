"""Cluster-based permutation testing for whole time-frequency maps."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
from scipy import ndimage
from scipy import stats

__all__ = ["cluster_perm_paired"]


def cluster_perm_paired(
    diff: np.ndarray, n_perm: int, alpha: float, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, float, float]]]:
    """Paired sign-flip cluster permutation test on an array (participants x ...).

    Clusters of neighbouring suprathreshold values are formed and their summed
    statistic is compared with the largest cluster obtainable by flipping signs
    at random. This controls the false-positive rate across the whole
    time-frequency map without testing every point separately
    (Maris & Oostenveld, 2007).

    Args:
        diff: Differences (condition A minus B), participants x ... .
        n_perm: Number of sign-flip permutations.
        alpha: Two-sided significance level for the cluster-forming threshold.
        seed: Random seed.

    Returns:
        ``(t_map, labels, table)``. ``table`` holds one row per cluster as
        ``(label, mass, p_value)``; positive labels mark increases and negative
        labels decreases.
    """
    n = diff.shape[0]
    rng = np.random.default_rng(seed)
    thr = stats.t.ppf(1 - alpha / 2, n - 1)

    def tmap(d: np.ndarray) -> np.ndarray:
        sd = d.std(axis=0, ddof=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(sd > 0, d.mean(axis=0) / (sd / np.sqrt(n)), 0.0)

    def clusters(t: np.ndarray) -> List[Tuple[int, np.ndarray, float]]:
        out: List[Tuple[int, np.ndarray, float]] = []
        for sign in (1, -1):
            lab, k = ndimage.label(sign * t > thr)
            out += [(sign, lab == i, float(np.abs(t[lab == i]).sum())) for i in range(1, k + 1)]
        return out

    t_obs = tmap(diff)
    obs = clusters(t_obs)
    null = np.zeros(n_perm)
    for i in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=(n,) + (1,) * (diff.ndim - 1))
        cl = clusters(tmap(diff * signs))
        null[i] = max([c[2] for c in cl], default=0.0)

    labels = np.zeros(t_obs.shape, int)
    table: List[Tuple[int, float, float]] = []
    for j, (sign, mask, mass) in enumerate(sorted(obs, key=lambda c: -c[2]), start=1):
        p = float((1 + np.sum(null >= mass)) / (n_perm + 1))
        labels[mask] = j * sign
        table.append((j * sign, mass, p))
    return t_obs, labels, table
