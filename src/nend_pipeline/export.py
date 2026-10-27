"""Writing results to disk: tables, figures and the run log."""
from __future__ import annotations

import json
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import Config
from .plots import plot_coupling, plot_eeg_session, plot_nirs_session, plot_pairs, plot_raw_session

__all__ = ["save_results", "jsonable"]


def jsonable(o: Any) -> Any:
    """Convert numpy/pandas values into something ``json.dump`` can write."""
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


def _save_table(df: pd.DataFrame, path: Path) -> None:
    if df is not None and not df.empty:
        df.to_csv(path, index=False)


def save_results(cfg: Config, results, out_dir: Optional[Path] = None, log=print) -> Path:
    """Write every table, figure and the run log to ``out_dir``.

    Args:
        cfg: The configuration used for the run.
        results: A :class:`~nend_pipeline.pipeline.Results` object.
        out_dir: Where to write. Defaults to ``derivatives/nend_pipeline`` under the
            study folder, or a local fallback.
        log: Callable for progress messages.

    Returns:
        The directory that was written to.
    """
    from .pipeline import resolve_output_dir

    out = Path(out_dir) if out_dir else resolve_output_dir(cfg)
    fig_dir = out / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    _save_table(results.audit, out / "marker_audit.csv")
    _save_table(results.nirs_quality, out / "nirs_quality.csv")
    _save_table(results.session_table, out / "session_table.csv")
    _save_table(results.group, out / "group_comparison.csv")
    _save_table(results.trial_level, out / "trial_level_comparison.csv")
    _save_table(results.coupling, out / "coupling.csv")
    _save_table(results.mismatch, out / "spreadsheet_mismatch.csv")

    eeg_rows = [
        rec["eeg_feats"]["trials"].assign(subject=sub, session=ses)
        for (sub, ses), rec in results.sessions.items()
        if "eeg_feats" in rec
    ]
    if eeg_rows:
        _save_table(pd.concat(eeg_rows, ignore_index=True), out / "eeg_trials.csv")

    nirs_rows = [
        rec["nirs_bins"].assign(subject=sub, session=ses)
        for (sub, ses), rec in results.sessions.items()
        if "nirs_bins" in rec
    ]
    if nirs_rows:
        _save_table(pd.concat(nirs_rows, ignore_index=True), out / "nirs_bins.csv")

    # ---- figures -------------------------------------------------------------------------
    shown = 0
    for (sub, ses), rec in results.sessions.items():
        title = f"{sub} {ses}"
        if rec.get("eeg") is not None and rec.get("markers") is not None:
            fig = plot_raw_session(rec, title, cfg)
            fig.savefig(fig_dir / f"{sub}_{ses}_raw.png", bbox_inches="tight")
            plt.close(fig)
        if "eeg_feats" in rec:
            fig = plot_eeg_session(rec, f"{title} - EEG", cfg)
            fig.savefig(fig_dir / f"{sub}_{ses}_eeg.png", bbox_inches="tight")
            plt.close(fig)
        if "nirs_proc" in rec:
            fig = plot_nirs_session(rec, title, cfg)
            fig.savefig(fig_dir / f"{sub}_{ses}_nirs.png", bbox_inches="tight")
            plt.close(fig)
        shown += 1

    if not results.session_table.empty:
        fig = plot_pairs(results.session_table, cfg)
        fig.savefig(fig_dir / "paired_outcomes.png", bbox_inches="tight")
        plt.close(fig)
        fig = plot_coupling(results.session_table, cfg)
        if fig is not None:
            fig.savefig(fig_dir / "coupling_session_level.png", bbox_inches="tight")
            plt.close(fig)

    # ---- run log ------------------------------------------------------------------------
    run_log: Dict[str, Any] = dict(
        created=datetime.now().isoformat(timespec="seconds"),
        package_version="0.1.0",
        versions=dict(
            python=sys.version.split()[0],
            platform=platform.platform(),
            numpy=np.__version__,
            pandas=pd.__version__,
        ),
        config=cfg.to_dict(),
        sessions=dict(
            analysed=[f"{k[0]}/{k[1]}" for k, v in results.sessions.items() if not v.get("skipped")],
            skipped={f"{k[0]}/{k[1]}": v["skipped"] for k, v in results.sessions.items() if v.get("skipped")},
        ),
        completeness=dict(
            n_sessions=len(results.sessions),
            n_analysed=sum(1 for v in results.sessions.values() if not v.get("skipped")),
            elapsed_s=results.elapsed_s,
        ),
        availability_spreadsheet=results.sheet_path,
    )
    with open(out / "run_log.json", "w", encoding="utf-8") as fh:
        json.dump(run_log, fh, indent=2, default=jsonable)

    log(f"Results written to {out}")
    return out
