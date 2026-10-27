"""Building result tables: per-session outcomes, the fasting contrast and EEG-fNIRS coupling."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import Config
from .stats import bh_fdr, paired_summary, spearman_ci, trial_level_summary

__all__ = [
    "OUTCOMES",
    "build_session_table",
    "group_comparison",
    "trial_level_table",
    "coupling_table",
]

#: Outcome name -> plain-language description.
OUTCOMES: Dict[str, str] = {
    "theta_front_db": "frontal theta (dB vs rest)",
    "alpha_front_db": "frontal alpha (dB vs rest)",
    "beta_front_db": "frontal beta (dB vs rest)",
    "faa_alpha": "frontal alpha asymmetry (ln R - ln L)",
    "hbo_pfc": "PFC ΔHbO (µM)",
    "hbr_pfc": "PFC ΔHbR (µM)",
    "hbo_lat_asym": "HbO asymmetry (R - L, µM)",
    "hbo_slope_per_min": "HbO slope (µM/min)",
}


def build_session_table(sessions: Dict[Tuple[str, str], Dict[str, Any]], cfg: Config) -> pd.DataFrame:
    """One row per participant and session, combining the EEG and fNIRS outcomes."""
    rows: List[Dict[str, Any]] = []
    for (sub, ses), rec in sessions.items():
        row: Dict[str, Any] = dict(subject=sub, session=ses)

        feats = rec.get("eeg_feats")
        if feats is not None:
            tr = feats["trials"]
            u = tr[tr["usable"]] if "usable" in tr else tr
            if len(u) >= cfg.min_trials:
                row.update(
                    theta_front_db=u["theta_front_db"].mean(),
                    alpha_front_db=u["alpha_front_db"].mean(),
                    beta_front_db=u["beta_front_db"].mean(),
                    faa_alpha=u["faa_alpha"].mean(),
                    n_eeg_trials=int(len(u)),
                )

        summary = rec.get("nirs_summary")
        if summary is not None:
            row.update(
                hbo_pfc=summary["pfc_hbo"],
                hbr_pfc=summary["pfc_hbr"],
                hbo_left=summary["left_hbo"],
                hbo_right=summary["right_hbo"],
                hbo_lat_asym=summary["hbo_lat_asym"],
                hbo_slope_per_min=summary["hbo_slope_per_min"],
                nirs_baseline_s=rec["nirs_proc"]["baseline_len_s"],
            )
        rows.append(row)
    return pd.DataFrame(rows)


def group_comparison(table: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Paired fasting-versus-non-fasting contrast for every outcome.

    Only the pre-specified primary outcomes are corrected for multiple
    comparisons; the rest are reported as exploratory.
    """
    A, B = cfg.condition_a, cfg.condition_b
    rows: List[Dict[str, Any]] = []
    for col, label in OUTCOMES.items():
        if col not in table.columns:
            continue
        wide = table.pivot(index="subject", columns="session", values=col)
        if A not in wide.columns or B not in wide.columns:
            continue
        s = paired_summary(wide[A].to_numpy(), wide[B].to_numpy(), cfg)
        rows.append(dict(outcome=col, description=label, primary=col in cfg.primary, **s))

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    prim = df["primary"].to_numpy()
    df["p_fdr"] = np.nan
    df.loc[prim, "p_fdr"] = bh_fdr(df.loc[prim, "p_wilcoxon"].to_numpy())
    return df


def trial_level_table(
    sessions: Dict[Tuple[str, str], Dict[str, Any]], cfg: Config
) -> pd.DataFrame:
    """Exploratory within-person, trial-level comparison for each participant separately."""
    rows: List[Dict[str, Any]] = []
    for sub in sorted({k[0] for k in sessions}):
        a = sessions.get((sub, cfg.condition_a), {})
        b = sessions.get((sub, cfg.condition_b), {})
        if "eeg_feats" not in a or "eeg_feats" not in b:
            continue
        ta, tb = a["eeg_feats"]["trials"], b["eeg_feats"]["trials"]
        for col in ("theta_front_db", "alpha_front_db", "faa_alpha"):
            if col not in ta or col not in tb:
                continue
            rows.append(
                dict(subject=sub, outcome=col, **trial_level_summary(ta[col], tb[col], cfg))
            )
    return pd.DataFrame(rows)


def coupling_table(
    sessions: Dict[Tuple[str, str], Dict[str, Any]], session_table: pd.DataFrame, cfg: Config
) -> pd.DataFrame:
    """EEG-fNIRS coupling, at the session level and (exploratory) by trial index.

    The session level does not require the two devices' clocks to be aligned.
    The trial-index level pairs the *k*-th EEG trial with the *k*-th fNIRS bin,
    which assumes both devices started the task together - something the files
    cannot confirm, so those rows are labelled as exploratory.
    """
    rows: List[Dict[str, Any]] = []
    pairs = (("theta_front_db", "hbo_pfc"), ("alpha_front_db", "hbo_pfc"), ("theta_front_db", "hbr_pfc"))
    if not session_table.empty:
        for x, y in pairs:
            if x in session_table.columns and y in session_table.columns:
                n, rho, lo, hi, p = spearman_ci(
                    session_table[x], session_table[y], cfg.n_boot, cfg.seed
                )
                rows.append(
                    dict(level="session", x=x, y=y, n=n, rho=rho, ci_lo=lo, ci_hi=hi, p=p)
                )

    for (sub, ses), rec in sessions.items():
        if "eeg_feats" in rec and "nirs_bins" in rec:
            e = rec["eeg_feats"]["trials"]["theta_front_db"].to_numpy()
            h = rec["nirs_bins"]["pfc_hbo"].to_numpy()
            k = min(len(e), len(h))
            n, rho, lo, hi, p = spearman_ci(e[:k], h[:k], cfg.n_boot, cfg.seed)
            rows.append(
                dict(
                    level=f"trial-index (exploratory) {sub} {ses}",
                    x="theta_front_db", y="pfc_hbo", n=n, rho=rho,
                    ci_lo=lo, ci_hi=hi, p=p,
                )
            )
    return pd.DataFrame(rows)
