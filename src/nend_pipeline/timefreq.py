"""Time-frequency analysis of the EEG (Morlet wavelets, expressed in dB vs rest)."""
from __future__ import annotations

from typing import Any, Dict, Optional

import mne
import numpy as np
import pandas as pd

from .config import Config

__all__ = ["eeg_ersp"]


def eeg_ersp(
    x: np.ndarray,
    windows: pd.DataFrame,
    feats: Dict[str, Any],
    cfg: Config,
    pre_s: float = 1.0,
    bin_s: float = 0.25,
) -> Optional[Dict[str, Any]]:
    """Morlet time-frequency power per trial, in dB relative to the resting baseline.

    This is the classic event-related spectral perturbation (ERS/ERD) view: it shows
    *when* within a task item a rhythm changes, not just *how much* on average
    (Pfurtscheller & Lopes da Silva, 1999).

    Args:
        x: Filtered EEG, channels x samples in microvolts.
        windows: Table of task windows (start/end in seconds).
        feats: Output of :func:`nend_pipeline.eeg.eeg_session_features`.
        cfg: Run configuration.
        pre_s: Seconds of pre-marker signal to include.
        bin_s: Width of the time bins in seconds.

    Returns:
        Dictionary with ``freqs``, ``times``, per-trial ``trials`` and the mean map,
        or ``None`` when nothing could be computed.
    """
    fs, chs = cfg.eeg_sfreq, list(cfg.eeg_channels)
    freqs = np.arange(4.0, 40.5, 1.0)
    pw = mne.time_frequency.tfr_array_morlet(
        x[None],
        sfreq=fs,
        freqs=freqs,
        n_cycles=freqs / 2.0,
        output="power",
        zero_mean=True,
        verbose=False,
    )[0]

    # Baseline spectrum, skipping the first second of the baseline window (filter edge).
    b0, b1 = feats["base_span"]
    i0, i1 = int(round((b0 + 1.0) * fs)), int(round(b1 * fs))
    if i1 - i0 < fs:
        return None
    bp = pw[:, :, i0:i1].mean(axis=-1)

    step = int(round(bin_s * fs))
    n_pts = int(round((pre_s + cfg.trial_s) * fs))
    maps = []
    for w in windows.itertuples():
        s = int(round((w.start - pre_s) * fs))
        if s < 0 or s + n_pts > pw.shape[-1]:
            continue
        tr = feats["trials"]
        row = tr.loc[tr["trial"] == w.trial]
        if row.empty:
            continue
        seg = 10 * np.log10(pw[:, :, s : s + n_pts] / bp[:, :, None])
        usable = (n_pts // step) * step
        if usable == 0:
            continue
        seg = seg[:, :, :usable].reshape(len(chs), len(freqs), -1, step).mean(axis=-1)
        for i, c in enumerate(chs):
            if row.iloc[0].get(f"clean_{c}", 0) < cfg.eeg_min_trial_windows:
                seg[i] = np.nan         # channel was too contaminated in this trial
        maps.append(seg)

    if not maps:
        return None
    arr = np.stack(maps)
    times = -pre_s + (np.arange(arr.shape[-1]) + 0.5) * bin_s
    return dict(
        freqs=freqs,
        times=times,
        trials=arr,
        mean=np.nanmean(arr, axis=0),
        n=len(maps),
    )
