"""EEG analysis: filtering, band power per trial and frontal metrics.

Reference points for the choices made here are listed in ``docs/METHODS.md``.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import mne
import numpy as np
import pandas as pd
from scipy import signal as sps

from .config import Config

__all__ = [
    "preprocess_eeg",
    "window_psd",
    "band_power",
    "eeg_session_features",
    "retention_at",
]


def preprocess_eeg(data_uv: np.ndarray, cfg: Config) -> np.ndarray:
    """Zero-phase FIR band-pass filter, with optional average referencing.

    Average referencing is **off** by default: it needs a reasonable number of
    electrodes to be meaningful, and with four dry electrodes it can distort
    frontal asymmetry (Dien, 1998).
    """
    x = mne.filter.filter_data(
        data_uv,
        cfg.eeg_sfreq,
        cfg.eeg_band[0],
        cfg.eeg_band[1],
        method="fir",
        fir_design="firwin",
        verbose=False,
    )
    if cfg.eeg_reference == "average":
        x = x - x.mean(axis=0, keepdims=True)
    return x


def window_psd(
    x: np.ndarray, t0: float, t1: float, cfg: Config, min_windows: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Mean power spectral density over clean windows inside a time span.

    The span is covered by overlapping Hann windows of ``welch_win_s``. A window is
    discarded for a channel when its peak-to-peak amplitude exceeds
    ``eeg_reject_uv`` (blink or movement) or falls below ``eeg_flat_uv`` (electrode
    off). Discarding bad *slices* rather than whole trials keeps far more usable
    data than whole-epoch rejection.

    Returns:
        ``(freqs, psd, n_clean, n_windows)``. ``psd`` is channels x frequencies in
        microvolt squared per hertz, NaN where too few clean windows were available.
    """
    fs = cfg.eeg_sfreq
    nper = int(round(cfg.welch_win_s * fs))
    freqs = np.fft.rfftfreq(nper, 1.0 / fs)
    psd = np.full((x.shape[0], len(freqs)), np.nan)
    n_clean = np.zeros(x.shape[0], dtype=int)

    i0, i1 = max(int(round(t0 * fs)), 0), min(int(round(t1 * fs)), x.shape[1])
    starts = np.arange(i0, i1 - nper + 1, nper // 2)
    if len(starts) == 0:
        return freqs, psd, n_clean, 0

    segs = np.stack([x[:, s : s + nper] for s in starts])          # windows x channels x samples
    swing = np.ptp(segs, axis=-1)
    ok = (swing <= cfg.eeg_reject_uv) & (swing >= cfg.eeg_flat_uv)
    _, pw = sps.welch(segs, fs=fs, window="hann", nperseg=nper, noverlap=0, axis=-1)
    for c in range(x.shape[0]):
        n_clean[c] = int(ok[:, c].sum())
        if n_clean[c] >= min_windows:
            psd[c] = pw[ok[:, c], c].mean(axis=0)
    return freqs, psd, n_clean, len(starts)


def band_power(freqs: np.ndarray, psd: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Power (microvolt squared) in the half-open band ``[lo, hi)`` for each channel."""
    m = (freqs >= lo) & (freqs < hi)
    return psd[..., m].sum(axis=-1) * (freqs[1] - freqs[0])


def _nanmean_pair(a: float, b: float) -> float:
    """Mean of two values, ignoring whichever is missing."""
    v = np.array([a, b], dtype=float)
    return float(np.nanmean(v)) if np.isfinite(v).any() else np.nan


def eeg_session_features(x: np.ndarray, windows: pd.DataFrame, cfg: Config) -> Dict[str, Any]:
    """Per-trial band power for one session, expressed relative to the resting baseline.

    The baseline is the ``baseline_s`` before the first marker (or ``eeg_baseline``
    when set). Each trial's power is also given in decibels relative to that
    baseline: +3 dB means twice the baseline power, -3 dB half.

    Returns:
        Dictionary with the per-trial ``trials`` table, the baseline spectrum and
        the per-trial spectra.
    """
    fs, chs = cfg.eeg_sfreq, list(cfg.eeg_channels)
    first = float(windows["start"].iloc[0])
    b0, b1 = cfg.eeg_baseline or (max(cfg.eeg_discard_start_s, first - cfg.baseline_s), first)
    freqs, base_psd, base_clean, base_windows = window_psd(
        x, b0, b1, cfg, cfg.eeg_min_baseline_windows
    )
    base = {b: band_power(freqs, base_psd, *rng) for b, rng in cfg.bands.items()}
    left, right = (chs.index(c) for c in cfg.frontal_channels)

    rows: list[Dict[str, Any]] = []
    task_psd: list[np.ndarray] = []
    for w in windows.itertuples():
        row: Dict[str, Any] = dict(
            trial=w.trial,
            code=w.code,
            start_s=w.start,
            complete=bool(w.start >= 0 and w.end * fs <= x.shape[1] + 1),
        )
        psd = np.full((len(chs), len(freqs)), np.nan)
        if row["complete"]:
            _, psd, nc, _ = window_psd(x, w.start, w.end, cfg, cfg.eeg_min_trial_windows)
            row.update({f"clean_{c}": int(nc[i]) for i, c in enumerate(chs)})
        task_psd.append(psd)

        for b, rng in cfg.bands.items():
            p = band_power(freqs, psd, *rng)
            with np.errstate(divide="ignore", invalid="ignore"):
                db = 10 * np.log10(p / base[b])
            for i, c in enumerate(chs):
                row[f"{b}_{c}_uv2"], row[f"{b}_{c}_db"] = p[i], db[i]
            row[f"{b}_front_db"] = _nanmean_pair(db[left], db[right])

        a = band_power(freqs, psd, *cfg.bands["alpha"])
        with np.errstate(divide="ignore", invalid="ignore"):
            row["faa_alpha"] = float(np.log(a[right]) - np.log(a[left]))
        row["usable"] = bool(np.isfinite(row.get("theta_front_db", np.nan)))
        rows.append(row)

    trials = pd.DataFrame(rows)
    stacked = np.stack(task_psd)
    task_mean = np.nanmean(stacked, axis=0) if np.isfinite(stacked).any() else stacked[0]
    return dict(
        trials=trials,
        freqs=freqs,
        base_psd=base_psd,
        base_clean=base_clean,
        base_windows=base_windows,
        base_span=(b0, b1),
        task_psd=task_mean,
        psd_by_trial=stacked,
    )


def retention_at(
    x: np.ndarray,
    windows: pd.DataFrame,
    cfg: Config,
    thresholds: Tuple[float, ...] = (100, 150, 200, 300, 500),
) -> pd.DataFrame:
    """How many trials keep both frontal channels at different rejection thresholds.

    Reporting this range lets the threshold be chosen from the data rather than
    assumed (cf. Jas et al., 2017, on rejection sensitivity).
    """
    chs = list(cfg.eeg_channels)
    idx = [chs.index(c) for c in cfg.frontal_channels]
    rows = []
    for thr in thresholds:
        c2 = cfg.replace(eeg_reject_uv=float(thr))
        kept = 0
        for w in windows.itertuples():
            if w.start < 0 or w.end * cfg.eeg_sfreq > x.shape[1] + 1:
                continue
            _, _, nc, _ = window_psd(x, w.start, w.end, c2, c2.eeg_min_trial_windows)
            kept += int(all(nc[i] >= c2.eeg_min_trial_windows for i in idx))
        rows.append(dict(reject_uv=thr, trials_kept=kept, of=len(windows)))
    return pd.DataFrame(rows)
