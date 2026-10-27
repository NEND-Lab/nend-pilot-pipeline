"""fNIRS analysis: motion correction, filtering, baseline correction and 5-second bins.

The Mendi file carries its own ``Bs``/``BE`` labels and runs on its own clock, so
fNIRS is anchored to **its own** baseline rather than to the EEG automarkers.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy import signal as sps

from .config import Config

__all__ = ["robust_z", "tddr", "process_nirs", "nirs_bins", "nirs_summary", "HB_KEYS"]

#: Haemoglobin series carried through the analysis.
HB_KEYS = ("left_hbo", "left_hbr", "right_hbo", "right_hbr", "pfc_hbo", "pfc_hbr")


def robust_z(x: np.ndarray) -> np.ndarray:
    """Median/MAD z-score, insensitive to the very outliers we want to detect."""
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med)) * 1.4826
    return np.zeros_like(x) if not np.isfinite(mad) or mad == 0 else (x - med) / mad


def tddr(y: np.ndarray, fs: float) -> np.ndarray:
    """Temporal Derivative Distribution Repair (Fishburn et al., 2019, NeuroImage 184:171-179).

    Down-weights sudden jumps caused by head movement without deleting data,
    which matters when each recording already contains only ~90 seconds.
    """
    y = np.asarray(y, float)
    mean_y = y.mean()
    if fs > 1.2:
        low = sps.sosfiltfilt(sps.butter(3, 0.5, btype="low", fs=fs, output="sos"), y - mean_y)
    else:
        low = y - mean_y
    high = (y - mean_y) - low

    d = np.diff(low)
    w, mu = np.ones_like(d), np.inf
    for _ in range(50):
        mu_prev = mu
        mu = np.sum(w * d) / np.sum(w)
        if abs(mu - mu_prev) < 1e-6 * max(np.std(d), 1e-12):
            break
        dev = np.abs(d - mu)
        sigma = 1.4826 * np.median(dev)
        if sigma == 0:
            break
        r = dev / (sigma * 4.685)
        w = ((1 - r ** 2) * (r < 1)) ** 2
    corrected = np.concatenate([[0.0], np.cumsum(w * (d - mu))])
    return corrected + high + mean_y


def process_nirs(raw: Dict[str, Any], cfg: Config) -> Dict[str, Any]:
    """Clean the haemoglobin signals and express them as change from rest.

    Steps: discard the start-up ramp, repair motion artefacts, low-pass filter at
    0.2 Hz, then subtract the mean over the file's own baseline period.

    Note:
        No high-pass is applied by default. These recordings are about 90 seconds
        long, and a 0.01 Hz high-pass would need a 100-second period, which would
        also remove the slow task response we are trying to measure.
    """
    fs, t = raw["fs"], raw["t"]
    flags: List[str] = []
    labels = raw["labels"] if cfg.nirs_use_labels else {}
    be = labels.get("BE")
    if be is None:
        be = cfg.baseline_s
        flags.append("NO_BE_LABEL(used baseline_s)")
    b0 = max(float(labels.get("Bs", 0.0)), 0.0) + cfg.nirs_discard_start_s
    keep = t >= cfg.nirs_discard_start_s
    moving = np.abs(robust_z(raw["head_movement"])) > cfg.nirs_motion_z

    out: Dict[str, np.ndarray] = {}
    for k, y in raw["sig"].items():
        yy = y.copy()
        bad = ~np.isfinite(yy)
        if bad.any():
            yy[bad] = np.interp(t[bad], t[~bad], yy[~bad])
        yy = yy[keep]
        if cfg.nirs_tddr:
            yy = tddr(yy, fs)
        lo, hi = cfg.nirs_band
        if hi is not None and lo is None:
            yy = sps.sosfiltfilt(sps.butter(3, hi, btype="low", fs=fs, output="sos"), yy)
        elif lo is not None and hi is not None:
            yy = sps.sosfiltfilt(sps.butter(3, [lo, hi], btype="band", fs=fs, output="sos"), yy)
        out[k] = yy

    tt = t[keep]
    bmask = (tt >= b0) & (tt <= be)
    base_len = float(be - b0)
    if bmask.sum() < cfg.nirs_min_baseline_s * fs:
        flags.append("SHORT_BASELINE")
    if base_len < 0.5 * cfg.baseline_s:
        flags.append(f"BASELINE_{base_len:.0f}s_vs_{cfg.baseline_s:.0f}s_protocol")

    delta = {k: v - (v[bmask].mean() if bmask.any() else np.nan) for k, v in out.items()}
    for kind in ("hbo", "hbr"):
        delta[f"pfc_{kind}"] = 0.5 * (delta[f"left_{kind}"] + delta[f"right_{kind}"])

    return dict(
        t=tt,
        delta=delta,
        fs=fs,
        baseline=(b0, float(be)),
        baseline_len_s=base_len,
        moving=moving[keep],
        move_frac=float(moving.mean()),
        be=float(be),
        duration_s=raw["duration_s"],
        flags=flags,
    )


def nirs_bins(proc: Dict[str, Any], cfg: Config) -> pd.DataFrame:
    """Mean change from baseline in consecutive 5-second bins after ``BE``.

    Each bin starts ``nirs_shift_s`` after its marker because blood flow lags
    neural activity by a few seconds (Buxton et al., 2004). A bin is dropped
    (NaN) when more than ``nirs_max_motion_frac`` of it is movement-flagged.
    """
    t, rows = proc["t"], []
    k, start = 0, proc["be"] + cfg.nirs_shift_s
    while start + cfg.trial_s <= t[-1] + 1e-9 and k < 200:
        m = (t >= start) & (t < start + cfg.trial_s)
        mfrac = float(proc["moving"][m].mean()) if m.any() else np.nan
        row = dict(bin=k + 1, start_s=start, rel_s=k * cfg.trial_s, motion_frac=mfrac)
        ok = m.sum() >= 0.8 * cfg.trial_s * proc["fs"] and mfrac <= cfg.nirs_max_motion_frac
        for key in HB_KEYS:
            row[key] = float(proc["delta"][key][m].mean()) if ok else np.nan
        rows.append(row)
        k, start = k + 1, start + cfg.trial_s
    return pd.DataFrame(rows)


def nirs_summary(bins: pd.DataFrame, cfg: Config) -> Dict[str, Any]:
    """Session-level fNIRS outcomes over a fixed-length block.

    Using the same ``nirs_block_s`` window in every session keeps sessions
    comparable even when their baselines differ in length.
    """
    n_need = int(round(cfg.nirs_block_s / cfg.trial_s))
    blk = bins[bins["rel_s"] < cfg.nirs_block_s] if len(bins) else bins
    n_ok = int(blk["pfc_hbo"].notna().sum()) if len(blk) else 0
    enough = n_ok >= int(0.8 * n_need)

    out: Dict[str, Any] = dict(nirs_bins_ok=n_ok, nirs_bins_needed=n_need)
    for key in HB_KEYS:
        out[key] = float(blk[key].mean()) if enough else np.nan

    out["hbo_slope_per_min"] = np.nan
    if enough and n_ok >= 4:
        v = blk.dropna(subset=["pfc_hbo"])
        out["hbo_slope_per_min"] = float(np.polyfit(v["rel_s"] / 60.0, v["pfc_hbo"], 1)[0])
    out["hbo_lat_asym"] = out["right_hbo"] - out["left_hbo"]      # right minus left
    return out
