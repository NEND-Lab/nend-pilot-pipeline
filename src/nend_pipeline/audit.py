"""Auditing the automarkers and loading the fNIRS recordings."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .config import Config
from .loaders import trial_windows

__all__ = ["audit_markers", "load_nirs_json"]


def audit_markers(markers, source: str, eeg: Dict[str, Any], cfg: Config) -> Dict[str, Any]:
    """Summarise automarker quality and list problems in ``flags``.

    Flag meanings:

    ``TOO_SHORT``
        The recording is too short to hold a baseline and a task window.
    ``NO_MARKERS``
        No automarkers could be read from either the TSV or the EEG file.
    ``ISI_IRREGULAR``
        Consecutive markers are not spaced by the configured trial length.
    ``N_WINDOW(S)_OUTSIDE_RECORDING``
        Some task windows run past the end of the recording.
    ``SHORT_BASELINE``
        Less than half the protocol baseline is available before the first marker.
    ``TOO_FEW_TRIALS``
        Fewer complete trials than ``min_trials`` (blocking).
    ``TSV_CSV_MISMATCH``
        The automarker file and the marker column inside the EEG file disagree.
    """
    flags: List[str] = []
    n = len(markers)
    w = trial_windows(markers, cfg) if n else None
    isi = np.diff(markers["onset"].to_numpy(float)) if n > 1 else np.array([])
    complete = int(((w["start"] >= 0) & (w["end"] <= eeg["duration_s"])).sum()) if n else 0
    base_len = min(float(w["start"].iloc[0]) - cfg.eeg_discard_start_s, cfg.baseline_s) if n else 0.0

    if eeg["duration_s"] < 3 * cfg.welch_win_s + cfg.trial_s:
        flags.append("TOO_SHORT")
    if n == 0:
        flags.append("NO_MARKERS")
    if len(isi) and np.max(np.abs(isi - cfg.trial_s)) > 0.25:
        flags.append("ISI_IRREGULAR")
    if n and complete < n:
        flags.append(f"{n - complete}_WINDOW(S)_OUTSIDE_RECORDING")
    if n and base_len < 0.5 * cfg.baseline_s:
        flags.append("SHORT_BASELINE")
    if n and complete < cfg.min_trials:
        flags.append("TOO_FEW_TRIALS")

    csv_t = eeg["csv_marker_times"]
    diff = np.nan
    if source == "tsv" and len(csv_t) and n:
        diff = float(np.median([np.min(np.abs(csv_t - o)) for o in markers["onset"]]))
        if diff > 0.25:
            flags.append("TSV_CSV_MISMATCH")

    return dict(
        markers=n,
        source=source,
        codes="; ".join(sorted(markers["code"].unique())) if n else "",
        first_onset_s=float(markers["onset"].iloc[0]) if n else np.nan,
        median_isi_s=float(np.median(isi)) if len(isi) else np.nan,
        isi_range_s=f"{isi.min():.2f}-{isi.max():.2f}" if len(isi) else "",
        recording_s=round(eeg["duration_s"], 1),
        complete_windows=complete,
        baseline_before_first_s=round(max(base_len, 0.0), 1),
        csv_markers=len(csv_t),
        tsv_vs_csv_median_diff_s=diff,
        max_gap_s=round(eeg["max_gap_s"], 3),
        notes="; ".join(eeg["notes"]),
        flags=",".join(flags),
    )


def load_nirs_json(path: Path, cfg: Config) -> Dict[str, Any]:
    """Mendi JSON -> HbO/HbR for both sensors in micromolar, with time and baseline labels."""
    with open(path, "r", encoding="utf-8") as fh:
        d = json.load(fh)

    sig: Dict[str, np.ndarray] = {}
    for side in ("left", "right"):
        vals = d[side]["values"]
        for kind in ("hbo", "hbr"):
            raw = [float(v.get(kind, v.get(kind.upper(), np.nan))) for v in vals]
            sig[f"{side}_{kind}"] = np.asarray(raw) * cfg.nirs_unit_scale

    n = len(sig["left_hbo"])
    dur = float(d.get("durationSeconds") or n / 10.5)
    t = np.linspace(0.0, dur, n)
    labels = {str(x.get("label")): float(x.get("timeSeconds")) for x in d.get("labels", []) if "label" in x}
    hm = np.asarray(d.get("headMovement", []), dtype=float)
    if hm.shape != (n,):
        hm = np.zeros(n)

    return dict(
        sig=sig,
        t=t,
        fs=(n - 1) / dur,
        labels=labels,
        head_movement=hm,
        duration_s=dur,
        n_samples=n,
        title=d.get("title", ""),
        file=path.name,
    )
