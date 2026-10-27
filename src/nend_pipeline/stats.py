"""Statistical helpers: effect sizes, bootstrap intervals, FDR and cluster permutation tests."""
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from scipy import stats

from .config import Config

__all__ = [
    "bh_fdr",
    "boot_ci",
    "paired_summary",
    "trial_level_summary",
    "spearman_ci",
]


def bh_fdr(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (NaN-safe).

    Controls the expected proportion of false discoveries among the tests in a
    family (Benjamini & Hochberg, 1995).
    """
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return out
    pv = p[ok]
    order = np.argsort(pv)
    adj = pv[order] * len(pv) / (np.arange(len(pv)) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    res = np.empty_like(pv)
    res[order] = np.clip(adj, 0, 1)
    out[ok] = res
    return out


def boot_ci(x: np.ndarray, stat=np.mean, n: int = 5000, seed: int = 42) -> Tuple[float, float]:
    """Percentile bootstrap 95 % confidence interval of a statistic of a 1-D sample."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    d = np.apply_along_axis(stat, 1, x[idx])
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def paired_summary(a: np.ndarray, b: np.ndarray, cfg: Config) -> Dict[str, Any]:
    """Paired comparison of condition A minus B across participants.

    Every participant contributes their own difference, so between-person
    differences cancel out. With fewer than ``min_pairs_inference`` complete
    pairs the result is descriptive only: no p-values are produced.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    d = (a - b)[ok]
    n = int(ok.sum())
    res: Dict[str, Any] = dict(
        n_pairs=n,
        mean_a=float(np.mean(a[ok])) if n else np.nan,
        mean_b=float(np.mean(b[ok])) if n else np.nan,
        mean_diff=float(d.mean()) if n else np.nan,
        dz=np.nan,
        ci_lo=np.nan,
        ci_hi=np.nan,
        p_wilcoxon=np.nan,
        p_ttest=np.nan,
    )
    if n >= cfg.min_pairs_inference:
        sd = d.std(ddof=1)
        res["dz"] = float(d.mean() / sd) if sd > 0 else np.nan
        res["ci_lo"], res["ci_hi"] = boot_ci(d, np.mean, cfg.n_boot, cfg.seed)
        if np.any(d != 0):
            res["p_wilcoxon"] = float(stats.wilcoxon(d, zero_method="wilcox").pvalue)
            res["p_ttest"] = float(stats.ttest_1samp(d, 0.0).pvalue)
    return res


def trial_level_summary(x_a: np.ndarray, x_b: np.ndarray, cfg: Config) -> Dict[str, Any]:
    """Exploratory within-person comparison of trials.

    Trials from one person are not independent observations, so this describes
    that person's data rather than testing a population effect.
    """
    x_a, x_b = np.asarray(x_a, float), np.asarray(x_b, float)
    x_a, x_b = x_a[np.isfinite(x_a)], x_b[np.isfinite(x_b)]
    res: Dict[str, Any] = dict(
        n_a=len(x_a), n_b=len(x_b), mean_a=np.nan, mean_b=np.nan, diff=np.nan,
        hedges_g=np.nan, ci_lo=np.nan, ci_hi=np.nan, p_mannwhitney=np.nan,
    )
    if len(x_a) < 3 or len(x_b) < 3:
        return res
    sp = np.sqrt(
        ((len(x_a) - 1) * x_a.var(ddof=1) + (len(x_b) - 1) * x_b.var(ddof=1))
        / (len(x_a) + len(x_b) - 2)
    )
    j = 1 - 3 / (4 * (len(x_a) + len(x_b)) - 9)
    rng = np.random.default_rng(cfg.seed)
    bs = [rng.choice(x_a, len(x_a)).mean() - rng.choice(x_b, len(x_b)).mean() for _ in range(cfg.n_boot)]
    res.update(
        mean_a=float(x_a.mean()), mean_b=float(x_b.mean()),
        diff=float(x_a.mean() - x_b.mean()),
        hedges_g=float(j * (x_a.mean() - x_b.mean()) / sp) if sp > 0 else np.nan,
        ci_lo=float(np.percentile(bs, 2.5)), ci_hi=float(np.percentile(bs, 97.5)),
        p_mannwhitney=float(stats.mannwhitneyu(x_a, x_b, alternative="two-sided").pvalue),
    )
    return res


def spearman_ci(
    x: Sequence[float], y: Sequence[float], n_boot: int = 5000, seed: int = 42
) -> Tuple[int, float, float, float, float]:
    """Spearman rho with a bootstrap 95 % interval. Returns (n, rho, lo, hi, p)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    if n < 4 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return n, np.nan, np.nan, np.nan, np.nan
    rho, p = stats.spearmanr(x, y)
    rng = np.random.default_rng(seed)
    bs: List[float] = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if np.ptp(x[i]) > 0 and np.ptp(y[i]) > 0:
            bs.append(stats.spearmanr(x[i], y[i])[0])
    lo, hi = (np.percentile(bs, [2.5, 97.5]) if len(bs) > 100 else (np.nan, np.nan))
    return n, float(rho), float(lo), float(hi), float(p)
