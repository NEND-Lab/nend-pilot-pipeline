"""Synthetic data generator shaped like the real NEND recordings.

Used by the test-suite so that the pipeline can be exercised end to end without
accessing any participant data.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

__all__ = ["make_eeg", "make_events", "make_nirs", "build_dataset"]

CHANNELS = ("TP9", "AF7", "AF8", "TP10")


def make_eeg(
    path: Path,
    duration_s: float = 102.5,
    fs: int = 256,
    alpha: float = 1.0,
    theta: float = 1.0,
    blink: bool = True,
    n_markers: int = 13,
    first_onset_s: float = 40.274,
    marker_interval_s: float = 5.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Write a Muse-style EEG CSV. Returns the marker onsets in seconds."""
    rng = rng or np.random.default_rng(1)
    n = int(duration_s * fs)
    t = np.arange(n) / fs
    cols: dict = {
        "sample_index": np.arange(n),
        "time_s": t + 1000.0,
        "timestamp_ms": t * 1000,
        "iso_time": "1970-01-01T00:00:00Z",
        "session_code": "synthetic",
        "protocol": "n-back",
        "stop_reason": "",
        "event_marker": 0,
    }
    for ch in CHANNELS:
        x = np.cumsum(rng.normal(0, 1, n)) * 0.5                    # slow drift
        x += 8 * alpha * np.sin(2 * np.pi * 10 * t + rng.uniform(0, 6))  # alpha rhythm
        x += 6 * theta * np.sin(2 * np.pi * 6 * t)                     # theta rhythm
        x += rng.normal(0, 4, n)
        if blink and ch in ("AF7", "AF8"):
            for b in (47.0, 63.0):
                i = int(b * fs)
                x[i : i + 40] += 400
        cols[f"{ch}_uV"] = x

    ev = np.round(first_onset_s + marker_interval_s * np.arange(n_markers) + rng.normal(0, 0.003, n_markers), 3)
    df = pd.DataFrame(cols)
    df.loc[np.clip((ev * fs).astype(int), 0, n - 1), "event_marker"] = 1
    df.to_csv(path, index=False)
    return ev


def make_events(path: Path, onsets: np.ndarray, code: str = "AUTO: T1_NEND") -> None:
    """Write an automarker TSV."""
    pd.DataFrame(
        {"onset": onsets, "duration": 0.0, "description": code, "source": "auto"}
    ).to_csv(path, sep="\t", index=False)


def make_nirs(
    path: Path,
    baseline_end_s: float = 10.0,
    amplitude: float = 1.0,
    duration_s: float = 90.0,
    n_samples: int = 943,
    rng: Optional[np.random.Generator] = None,
) -> None:
    """Write a Mendi-style fNIRS JSON."""
    rng = rng or np.random.default_rng(2)
    t = np.linspace(0, duration_s, n_samples)
    ramp = 1 - np.exp(-t / 1.0)                                  # start-up transient

    def signal(amp: float, lag: float) -> np.ndarray:
        rise = np.where(t > baseline_end_s + lag, 1 - np.exp(-(t - baseline_end_s - lag) / 8), 0)
        return -3e-6 * np.exp(-t / 0.8) + amp * 1e-6 * rise * ramp

    def series(a: float, b: float) -> List[dict]:
        hb_o, hb_r = signal(a, 4.0), signal(-b, 4.0)
        return [
            {"hbo": float(o), "hbr": float(r)}
            for o, r in zip(hb_o + rng.normal(0, 3e-8, n_samples),
                            hb_r + rng.normal(0, 3e-8, n_samples) + 2e-7 * np.sin(2 * np.pi * t))
        ]

    head = np.abs(rng.normal(3, 0.5, n_samples))
    head[400:410] = 40
    payload = {
        "title": "synthetic",
        "durationSeconds": duration_s,
        "left": {"values": series(amplitude, 0.3 * amplitude)},
        "right": {"values": series(amplitude * 1.1, 0.3 * amplitude)},
        "headMovement": head.tolist(),
        "heartRate": [],
        "labels": [
            {"timeSeconds": 0, "label": "Bs"},
            {"timeSeconds": baseline_end_s, "label": "BE"},
        ],
        "rawFrames": [],
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)


def build_dataset(root: Path, n_subjects: int = 7, seed: int = 1) -> Path:
    """Create a synthetic study folder. Returns the ``NEND-pilot`` path.

    Participant 1 has an automarker file with only a header (so the CSV fallback
    is exercised) and only one session (so the pairing logic is exercised).
    """
    rng = np.random.default_rng(seed)
    study = root / "NEND-pilot"
    for i in range(1, n_subjects + 1):
        sub = f"sub-NEND{i:03d}"
        for ses, factor in (("fasting", 1.4), ("nonfasting", 1.0)):
            if i == 1 and ses == "nonfasting":
                continue
            d = study / sub / f"ses-{ses}"
            (d / "eeg").mkdir(parents=True, exist_ok=True)
            (d / "nirs").mkdir(parents=True, exist_ok=True)
            tag = f"{sub}_ses-{ses}_task-2back_run-01"
            onsets = make_eeg(
                d / "eeg" / f"{tag}_eeg.csv",
                theta=factor * rng.uniform(0.8, 1.2),
                alpha=rng.uniform(0.8, 1.2),
                rng=rng,
            )
            if i == 1:
                (d / "eeg" / f"{tag}_events.tsv").write_text(
                    "onset\tduration\tdescription\tsource\n", encoding="utf-8"
                )
            else:
                make_events(d / "eeg" / f"{tag}_events.tsv", onsets)
            make_nirs(
                d / "nirs" / f"{tag}_nirs.json",
                baseline_end_s=10.0 if ses == "fasting" else 30.0,
                amplitude=factor * rng.uniform(0.8, 1.2),
                rng=rng,
            )
    return study
