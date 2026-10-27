"""Orchestration: run the whole analysis for a cohort and collect the results."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .audit import audit_markers, load_nirs_json
from .config import Config
from .eeg import eeg_session_features, preprocess_eeg, retention_at
from .inventory import build_inventory, compare_with_sheet, read_availability
from .loaders import (
    is_blocking,
    load_eeg_csv,
    load_events_tsv,
    select_markers,
    trial_windows,
)
from .nirs import nirs_bins, nirs_summary, process_nirs
from .results import (
    build_session_table,
    coupling_table,
    group_comparison,
    trial_level_table,
)
from .timefreq import eeg_ersp

__all__ = ["Results", "run_analysis", "resolve_output_dir"]


@dataclass
class Results:
    """Everything produced by one analysis run."""

    config: Config
    inventory: pd.DataFrame
    audit: pd.DataFrame
    nirs_quality: pd.DataFrame
    session_table: pd.DataFrame
    group: pd.DataFrame
    trial_level: pd.DataFrame
    coupling: pd.DataFrame
    mismatch: pd.DataFrame
    sessions: Dict[Tuple[str, str], Dict[str, Any]] = field(default_factory=dict)
    sheet_path: Optional[str] = None
    elapsed_s: float = 0.0


def run_analysis(cfg: Config, log=print) -> Results:
    """Analyse every session in the inventory and return all result tables.

    A session that fails is recorded with its reason and skipped, so one bad file
    cannot stop the cohort.

    Args:
        cfg: Run configuration.
        log: Callable used for progress messages.

    Returns:
        A :class:`Results` object with every table plus the per-session data.
    """
    t0 = time.time()
    log(f"Study folder : {cfg.resolved_bids_root()}")
    inv = build_inventory(cfg)
    sheet = read_availability(cfg)
    mismatch = compare_with_sheet(inv, sheet)
    log(f"Participants : {inv['subject'].nunique()} | sessions: {len(inv)}")
    if sheet is None:
        log("No availability spreadsheet found - cross-check skipped.")
    elif mismatch.empty:
        log(f"Spreadsheet cross-check ({sheet.attrs['path']}): no disagreements.")
    else:
        log(f"Spreadsheet cross-check: {len(mismatch)} disagreement(s).")

    sessions: Dict[Tuple[str, str], Dict[str, Any]] = {}
    audit_rows: List[Dict[str, Any]] = []

    for r in inv.itertuples():
        key = (r.subject, r.session)
        rec: Dict[str, Any] = dict(subject=r.subject, session=r.session, skipped=None)

        # ---- EEG and automakers -----------------------------------------------------------
        try:
            if not r.has_eeg:
                raise FileNotFoundError("no EEG csv")
            eeg = load_eeg_csv(r.eeg_csv, cfg)
            tsv = load_events_tsv(r.events_tsv)
            markers, source = select_markers(tsv, eeg["csv_marker_times"], cfg)
            info = audit_markers(markers, source, eeg, cfg)
            rec.update(
                eeg=eeg, markers=markers,
                windows=trial_windows(markers, cfg) if len(markers) else None,
                audit=info,
            )
            if is_blocking(info["flags"]):
                rec["skipped"] = f"EEG blocked by: {info['flags']}"
        except Exception as exc:                       # one bad file must not stop the cohort
            rec["skipped"] = f"EEG load failed: {type(exc).__name__}: {exc}"
            info = dict(flags="LOAD_FAILED")

        # ---- fNIRS -------------------------------------------------------------------------
        rec["nirs_raw"] = None
        if r.has_nirs:
            try:
                rec["nirs_raw"] = load_nirs_json(r.nirs_json, cfg)
            except Exception as exc:
                rec["nirs_error"] = f"{type(exc).__name__}: {exc}"

        sessions[key] = rec
        audit_rows.append(
            dict(
                subject=r.subject, session=r.session,
                **{k: v for k, v in info.items() if k != "notes"},
                nirs_s=rec["nirs_raw"]["duration_s"] if rec["nirs_raw"] else np.nan,
                nirs_labels=str(rec["nirs_raw"]["labels"]) if rec["nirs_raw"] else "",
                skipped=rec["skipped"] or "",
            )
        )
        log(f"  loaded {r.subject} {r.session}" + (f"  [skipped: {rec['skipped']}]" if rec["skipped"] else ""))

    _run_eeg_features(sessions, cfg, log)
    nirs_q = _run_nirs_features(sessions, cfg, log)

    session_table = build_session_table(sessions, cfg)
    log(f"Sessions with usable EEG: {sum(1 for v in sessions.values() if v.get('eeg_feats'))}")
    log(f"Complete pairs for the group contrast: {_complete_pairs(session_table, cfg)}")

    return Results(
        config=cfg,
        inventory=inv,
        audit=pd.DataFrame(audit_rows),
        nirs_quality=pd.DataFrame(nirs_q),
        session_table=session_table,
        group=group_comparison(session_table, cfg) if not session_table.empty else pd.DataFrame(),
        trial_level=trial_level_table(sessions, cfg),
        coupling=coupling_table(sessions, session_table, cfg),
        mismatch=mismatch,
        sessions=sessions,
        sheet_path=sheet.attrs.get("path") if sheet is not None else None,
        elapsed_s=round(time.time() - t0, 1),
    )


def resolve_output_dir(cfg: Config) -> Path:
    """Pick a writable output folder, falling back to a local path if needed.

    Order of preference: the configured ``output_dir``, then ``derivatives`` inside
    the study folder, then ``/content/nend_outputs``, then the current directory.
    """
    candidates: List[Path] = []
    if cfg.output_dir:
        candidates.append(Path(cfg.output_dir))
    try:
        candidates.append(Path(cfg.resolved_bids_root()) / "derivatives" / "nend_pipeline")
    except ValueError:
        pass
    candidates += [Path("/content/nend_outputs"), Path.cwd() / "nend_outputs"]

    for c in candidates:
        try:
            (c / "figures").mkdir(parents=True, exist_ok=True)
            probe = c / ".write_test"
            probe.write_text("ok")
            probe.unlink()
            return c
        except OSError:
            continue
    raise OSError("No writable output folder found; set --out explicitly.")


def _run_eeg_features(sessions: Dict[Tuple[str, str], Dict[str, Any]], cfg: Config, log) -> None:
    """Compute band power, retention and time-frequency maps for each usable session."""
    for (sub, ses), rec in sessions.items():
        if rec["skipped"] or rec.get("eeg") is None:
            continue
        try:
            x = preprocess_eeg(rec["eeg"]["data_uv"], cfg)
            feats = eeg_session_features(x, rec["windows"], cfg)
            rec.update(
                eeg_x=x,
                eeg_feats=feats,
                retention=retention_at(x, rec["windows"], cfg),
            )
            rec["ersp"] = eeg_ersp(x, rec["windows"], feats, cfg)
            n_use = int(feats["trials"]["usable"].sum())
            rec["n_eeg_usable"] = n_use
            if n_use < cfg.min_trials:
                rec["skipped_eeg_summary"] = f"only {n_use} usable EEG trials"
        except Exception as exc:
            rec["skipped"] = f"EEG analysis failed: {type(exc).__name__}: {exc}"
            log(f"  EEG analysis failed for {sub} {ses}: {rec['skipped']}")


def _run_nirs_features(
    sessions: Dict[Tuple[str, str], Dict[str, Any]], cfg: Config, log
) -> List[Dict[str, Any]]:
    """Process the fNIRS recordings and collect the per-session quality rows."""
    quality: List[Dict[str, Any]] = []
    for (sub, ses), rec in sessions.items():
        raw = rec.get("nirs_raw")
        if raw is None:
            continue
        try:
            proc = process_nirs(raw, cfg)
            bins = nirs_bins(proc, cfg)
            summ = nirs_summary(bins, cfg)
        except Exception as exc:
            rec["nirs_error"] = f"{type(exc).__name__}: {exc}"
            log(f"  fNIRS analysis failed for {sub} {ses}: {rec['nirs_error']}")
            continue
        rec.update(nirs_proc=proc, nirs_bins=bins, nirs_summary=summ)
        quality.append(
            dict(
                subject=sub, session=ses,
                duration_s=round(proc["duration_s"], 1), fs_hz=round(proc["fs"], 2),
                baseline_len_s=round(proc["baseline_len_s"], 1),
                move_frac=round(proc["move_frac"], 3),
                bins_ok=summ["nirs_bins_ok"], bins_needed=summ["nirs_bins_needed"],
                flags=",".join(proc["flags"]),
            )
        )
    return quality


def _complete_pairs(session_table: pd.DataFrame, cfg: Config) -> int:
    """Number of participants with usable data in both conditions."""
    if session_table.empty or "theta_front_db" not in session_table.columns:
        return 0
    wide = session_table.pivot_table(
        index="subject", columns="session", values="theta_front_db", aggfunc="mean"
    )
    if not {cfg.condition_a, cfg.condition_b} <= set(wide.columns):
        return 0
    return int(wide[[cfg.condition_a, cfg.condition_b]].notna().all(axis=1).sum())
