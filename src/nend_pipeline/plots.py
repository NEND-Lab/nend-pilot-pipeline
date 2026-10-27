"""Figure generation for the NEND pipeline."""
from __future__ import annotations

from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import Config

#: Colours used throughout the figures.
COLORS = {"fasting": "#d62728", "nonfasting": "#1f77b4", "diff": "#6a3d9a"}

__all__ = [
    "COLORS",
    "plot_raw_session",
    "plot_eeg_session",
    "plot_nirs_session",
    "plot_pairs",
    "plot_coupling",
]

def plot_nirs_session(rec: Dict[str, Any], title: str, cfg: Config):
    """Change-from-baseline haemoglobin traces with the analysis bins marked."""
    proc, bins = rec["nirs_proc"], rec["nirs_bins"]
    fig, axes = plt.subplots(2, 1, figsize=(13, 6.5), sharex=True)
    for ax, kind, name in ((axes[0], "hbo", "dHbO (uM)"), (axes[1], "hbr", "dHbR (uM)")):
        ax.plot(
            proc["t"], proc["delta"][f"left_{kind}"],
            color="#d62728" if kind == "hbo" else "#1f77b4", lw=1.2, label="left",
        )
        ax.plot(
            proc["t"], proc["delta"][f"right_{kind}"],
            color="#ff9896" if kind == "hbo" else "#aec7e8", lw=1.2, label="right",
        )
        ax.axvspan(*proc["baseline"], color="gray", alpha=0.2, label="baseline")
        for s in bins["start_s"]:
            ax.axvline(s, color="k", alpha=0.12, lw=0.6)
        ax.axhline(0, color="k", lw=0.6)
        ax.set(ylabel=name)
        ax.legend(loc="upper left", ncol=3, fontsize=8)
    axes[0].set_title(
        f"{title} - change from resting baseline "
        f"(5-s bins shifted by {cfg.nirs_shift_s:g} s lag)"
    )
    axes[1].set_xlabel("time (s, fNIRS clock)")
    fig.tight_layout()
    return fig


def plot_pairs(table: pd.DataFrame, cfg: Config):
    """Paired lines from each participant: fasting versus non-fasting."""
    from .results import OUTCOMES

    cols = [c for c in ("theta_front_db", "alpha_front_db", "faa_alpha", "hbo_pfc", "hbr_pfc")
            if c in table.columns]
    fig, axes = plt.subplots(1, len(cols), figsize=(3.2 * len(cols), 4), squeeze=False)
    for ax, c in zip(axes[0], cols):
        wide = table.pivot(index="subject", columns="session", values=c)
        for _, r in wide.iterrows():
            ax.plot(
                [0, 1], [r.get(cfg.condition_a), r.get(cfg.condition_b)],
                "-o", color="gray", alpha=0.6,
            )
        ax.set_xticks([0, 1], [cfg.condition_a, cfg.condition_b])
        ax.set_title(OUTCOMES[c], fontsize=9)
    fig.suptitle("Each line = one participant (fasting -> non-fasting)")
    fig.tight_layout()
    return fig


def plot_coupling(table: pd.DataFrame, cfg: Config):
    """Session-level scatter of frontal theta against prefrontal oxygenation."""
    if table.empty:
        return None
    d = table.dropna(subset=["theta_front_db", "hbo_pfc"])
    if len(d) < 3:
        return None
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for ses, g in d.groupby("session"):
        ax.scatter(
            g["theta_front_db"], g["hbo_pfc"], s=60,
            color=COLORS.get(ses, "gray"), label=ses, edgecolor="k",
        )
    ax.set(
        xlabel="frontal theta (dB vs rest, EEG)",
        ylabel="PFC dHbO (uM, fNIRS)",
        title="Session-level coupling",
    )
    ax.legend()
    fig.tight_layout()
    return fig


def plot_raw_session(rec: Dict[str, Any], title: str, cfg: Config):
    """Raw EEG channels with markers, and the raw fNIRS traces with their labels."""
    has_n = rec.get("nirs_raw") is not None
    fig, axes = plt.subplots(
        2 if has_n else 1, 1, figsize=(13, 6.5 if has_n else 3.8), squeeze=False
    )
    ax = axes[0, 0]
    eeg = rec["eeg"]
    x = eeg["data_uv"] - np.median(eeg["data_uv"], axis=1, keepdims=True)
    for i, c in enumerate(cfg.eeg_channels):
        ax.plot(eeg["t"], x[i] + i * 300, lw=0.5, label=c)
    for o in rec["markers"]["onset"]:
        ax.axvline(o, color="k", ls=":", lw=0.8, alpha=0.7)
    ax.set(
        title=f"{title} - raw EEG (offset for display) with automarkers (dotted)",
        xlabel="time (s)",
        ylabel="uV (offset)",
    )
    ax.legend(loc="upper left", ncol=4, fontsize=8)

    if has_n:
        nr = rec["nirs_raw"]
        ax2 = axes[1, 0]
        ax2.plot(nr["t"], nr["sig"]["left_hbo"], color="#d62728", lw=1, label="HbO left")
        ax2.plot(nr["t"], nr["sig"]["right_hbo"], color="#ff9896", lw=1, label="HbO right")
        ax2.plot(nr["t"], nr["sig"]["left_hbr"], color="#1f77b4", lw=1, label="HbR left")
        ax2.plot(nr["t"], nr["sig"]["right_hbr"], color="#aec7e8", lw=1, label="HbR right")
        for lab, tt in nr["labels"].items():
            ax2.axvline(tt, color="gray", ls="--")
            ax2.text(tt, ax2.get_ylim()[1], f" {lab}", va="top", fontsize=8)
        ax2.set(
            title="raw fNIRS (uM; the first seconds are a start-up ramp) - dashed = file labels",
            xlabel="time (s, own clock)",
            ylabel="uM",
        )
        ax2.legend(loc="upper left", ncol=4, fontsize=8)
    fig.tight_layout()
    return fig


def plot_eeg_session(rec: Dict[str, Any], title: str, cfg: Config):
    """Spectrum, per-trial band power, time-frequency maps and frontal asymmetry."""
    feats = rec["eeg_feats"]
    f, tr = feats["freqs"], feats["trials"]
    chs = list(cfg.eeg_channels)

    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 3)

    ax = fig.add_subplot(gs[0, 0])
    for i, c in enumerate(chs):
        ax.semilogy(f, feats["base_psd"][i], color="gray", alpha=0.5, lw=1)
        ax.semilogy(f, feats["task_psd"][i], lw=1.4, label=c)
    ax.set(xlim=(1, 40), xlabel="Hz", ylabel="uV^2/Hz", title="Spectrum: rest (grey) vs task")
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[0, 1:])
    for b, col in (("theta", "#2ca02c"), ("alpha", "#ff7f0e"), ("beta", "#9467bd")):
        ax.plot(tr["trial"], tr[f"{b}_front_db"], marker="o", color=col, label=f"{b} (frontal)")
    ax.axhline(0, color="k", lw=0.8)
    ax.set(
        xlabel="trial (marker number)",
        ylabel="dB vs resting baseline",
        title="Frontal band power per 5-s trial",
    )
    ax.legend(fontsize=8)

    ers = rec.get("ersp")
    for j, c in enumerate(cfg.frontal_channels):
        ax = fig.add_subplot(gs[1, j])
        if ers is not None:
            ci = chs.index(c)
            im = ax.pcolormesh(
                ers["times"], ers["freqs"], ers["mean"][ci],
                cmap="RdBu_r", vmin=-4, vmax=4, shading="auto",
            )
            ax.axvline(0, color="k", ls="--")
            fig.colorbar(im, ax=ax, label="dB")
        ax.set(title=f"ERSP {c} (n={ers['n'] if ers else 0})", xlabel="s from marker", ylabel="Hz")

    ax = fig.add_subplot(gs[1, 2])
    ax.plot(tr["trial"], tr["faa_alpha"], marker="s", color=COLORS["diff"])
    ax.axhline(0, color="k", lw=0.8)
    ax.set(xlabel="trial", ylabel="ln(AF8) - ln(AF7)", title="Frontal alpha asymmetry")

    fig.suptitle(title)
    fig.tight_layout()
    return fig
