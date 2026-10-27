"""Loading the raw EEG and automarker files."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import Config

__all__ = [
    "load_eeg_csv",
    "load_events_tsv",
    "select_markers",
    "trial_windows",
    "BLOCKING_FLAGS",
    "is_blocking",
]

#: Flags that make a session unusable for analysis.
BLOCKING_FLAGS = ("TOO_SHORT", "NO_MARKERS", "TOO_FEW_TRIALS")


def load_eeg_csv(path: Path, cfg: Config) -> Dict[str, Any]:
    """Muse CSV -> microvolt array (channels x samples) plus the time vector.

    The time vector is re-zeroed to the first sample so that automarker onsets,
    which are relative to the start of the recording, line up with it.
    """
    df = pd.read_csv(path)
    cols = list(cfg.eeg_channels.values())
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"EEG columns missing in {path.name}: {missing}. Found: {list(df.columns)}")

    notes: List[str] = []
    data = df[cols].to_numpy(dtype=float).T
    n = data.shape[1]

    if "time_s" in df.columns:
        t = df["time_s"].to_numpy(dtype=float) - float(df["time_s"].iloc[0])
    else:
        t = np.arange(n) / cfg.eeg_sfreq
        notes.append("no 'time_s' column: nominal sampling rate used")

    if n < 2 or np.any(np.diff(t) <= 0):
        notes.append("time_s not strictly increasing: nominal sampling rate used")
        t = np.arange(n) / cfg.eeg_sfreq

    nan_frac = float(np.isnan(data).mean())
    if nan_frac > 0:
        for i in range(data.shape[0]):
            bad = np.isnan(data[i])
            if bad.all():
                raise ValueError(f"Channel {list(cfg.eeg_channels)[i]} is entirely NaN in {path.name}")
            data[i, bad] = np.interp(t[bad], t[~bad], data[i, ~bad])
        notes.append(f"{nan_frac:.1%} NaN samples linearly interpolated")

    csv_markers = np.array([])
    if "event_marker" in df.columns:
        s = df["event_marker"].astype(str).str.strip().str.lower()
        csv_markers = t[(~s.isin(["", "0", "0.0", "nan", "none", "false"])).to_numpy()]

    return dict(
        data_uv=data,
        t=t,
        duration_s=float(t[-1]) if n else 0.0,
        n_samples=n,
        max_gap_s=float(np.diff(t).max()) if n > 1 else 0.0,
        nan_frac=nan_frac,
        csv_marker_times=csv_markers,
        notes=notes,
        file=path.name,
    )


def load_events_tsv(path: Optional[Path]) -> pd.DataFrame:
    """Automarker TSV -> table with ``onset`` (s) and ``code`` columns.

    A missing or empty file yields an empty table rather than an error.
    """
    empty = pd.DataFrame(columns=["onset", "code"])
    if path is None or not Path(path).is_file():
        return empty
    try:
        df = pd.read_csv(path, sep="\t")
        if df.shape[1] == 1:
            df = pd.read_csv(path, sep=None, engine="python")
    except pd.errors.EmptyDataError:
        return empty
    if df.empty or "onset" not in df.columns:
        return empty
    code_col = next(
        (c for c in ("description", "trial_type", "value", "code", "label") if c in df.columns),
        None,
    )
    out = pd.DataFrame(
        {
            "onset": pd.to_numeric(df["onset"], errors="coerce"),
            "code": df[code_col].astype(str) if code_col else "marker",
        }
    )
    return out.dropna(subset=["onset"]).sort_values("onset").reset_index(drop=True)


def select_markers(tsv: pd.DataFrame, csv_times: np.ndarray, cfg: Config) -> Tuple[pd.DataFrame, str]:
    """Choose the marker list: the automarker file first, the EEG ``event_marker`` column as fallback."""
    m, source = tsv.copy(), "tsv"
    if cfg.marker_regex and len(m):
        m = m[m["code"].str.contains(cfg.marker_regex, regex=True)]
    if m.empty and cfg.events_fallback_to_csv and len(csv_times):
        m, source = pd.DataFrame({"onset": csv_times, "code": "CSV:event_marker"}), "csv"
    if m.empty:
        source = "none"
    m = m.reset_index(drop=True)
    m["trial"] = np.arange(1, len(m) + 1)
    return m, source


def trial_windows(markers: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Start and end (s) of each task window, following the ``marker_is_trial_start`` setting."""
    on = markers["onset"].to_numpy(float)
    start = on if cfg.marker_is_trial_start else on - cfg.trial_s
    return pd.DataFrame(
        {
            "trial": markers["trial"].to_numpy(),
            "start": start,
            "end": start + cfg.trial_s,
            "code": markers["code"].to_numpy(),
        }
    )


def is_blocking(flags: str) -> bool:
    """Whether a flag string contains a problem that makes a session unusable."""
    return any(b in flags for b in BLOCKING_FLAGS)
