"""Locating the study data and cross-checking it against the availability spreadsheet."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import Config

__all__ = [
    "norm_subject",
    "norm_session",
    "build_inventory",
    "read_availability",
    "compare_with_sheet",
]


def norm_subject(name: str) -> str:
    """``NEND002`` and ``sub-NEND002`` both become ``sub-NEND002``."""
    name = str(name).strip()
    return name if name.startswith("sub-") else f"sub-{name}"


def norm_session(folder_name: str) -> str:
    """Map ``ses-fasting`` / ``ses-nonfasting`` / ``ses-non-fasting`` to two condition labels."""
    n = re.sub(r"[^a-z]", "", folder_name.lower().replace("ses-", ""))
    return "nonfasting" if n.startswith("non") or n in {"fed", "postprandial"} else "fasting"


def first_match(folder: Path, pattern: str) -> Tuple[Optional[Path], int]:
    """First matching file in ``folder`` and how many matches there were."""
    hits = sorted(folder.glob(pattern)) if folder.is_dir() else []
    return (hits[0] if hits else None), len(hits)


def build_inventory(cfg: Config) -> pd.DataFrame:
    """One row per participant and session, listing the EEG, automarker and fNIRS files.

    Raises:
        FileNotFoundError: if the study folder is missing or holds no ``sub-*`` folders.
        ValueError: if a requested participant is absent (listing what *is* available).
    """
    root = Path(cfg.resolved_bids_root())
    if not root.is_dir():
        raise FileNotFoundError(
            f"Study folder not found: {root}\n"
            "-> Check the path, or set the NEND_BIDS_ROOT environment variable."
        )
    found = sorted(p for p in root.glob("sub-*") if p.is_dir())
    if not found:
        listing = ", ".join(sorted(p.name for p in root.iterdir())[:15])
        raise FileNotFoundError(f"No 'sub-*' folders inside {root}. It contains: {listing or '(empty)'}")

    wanted = None if cfg.subjects is None else {norm_subject(s) for s in cfg.subjects}
    if wanted is not None:
        absent = wanted - {p.name for p in found}
        if absent:
            raise ValueError(
                f"Requested participant(s) not found: {sorted(absent)}. Available: {[p.name for p in found]}"
            )

    rows: List[Dict[str, Any]] = []
    for sub in found:
        if wanted is not None and sub.name not in wanted:
            continue
        seen = set()
        for ses in sorted(p for p in sub.glob("ses-*") if p.is_dir()):
            label = norm_session(ses.name)
            if label in seen:
                continue
            seen.add(label)
            eeg_csv, n_eeg = first_match(ses / "eeg", "*_eeg.csv")
            tsv, _ = first_match(ses / "eeg", "*_events.tsv")
            nirs, n_nirs = first_match(ses / "nirs", "*_nirs.json")
            rows.append(
                dict(
                    subject=sub.name,
                    session=label,
                    eeg_csv=eeg_csv,
                    events_tsv=tsv,
                    nirs_json=nirs,
                    n_eeg_files=n_eeg,
                    n_nirs_files=n_nirs,
                )
            )
    inv = pd.DataFrame(rows)
    if inv.empty:
        raise FileNotFoundError("Participant folders were found but none contains a 'ses-*' folder.")
    inv["has_eeg"] = inv["eeg_csv"].notna()
    inv["has_nirs"] = inv["nirs_json"].notna()
    return inv


def read_availability(cfg: Config) -> Optional[pd.DataFrame]:
    """Parse the ``Available datasets.xlsx`` sheet, if it can be found.

    Returns ``None`` when the sheet is absent, so the pipeline works without it.
    """
    root = Path(cfg.resolved_bids_root())
    places = [Path(cfg.availability_xlsx)] if cfg.availability_xlsx else []
    places += [d / "Available datasets.xlsx" for d in (root, root.parent, Path.cwd())]
    path = next((p for p in places if p.is_file()), None)
    if path is None:
        return None
    raw = pd.read_excel(path, header=None, engine="openpyxl")
    ids = raw.iloc[:, 0].astype(str).str.strip()
    body = raw[ids.str.match(r"^NEND\d+")]

    def flag(col: int) -> np.ndarray:
        return body.iloc[:, col].astype(str).str.strip().str.lower().eq("seen").to_numpy()

    out = pd.DataFrame(
        {
            "subject": ("sub-" + ids[body.index]).to_numpy(),
            "fasting_nirs": flag(1),
            "fasting_eeg": flag(2),
            "nonfasting_nirs": flag(3),
            "nonfasting_eeg": flag(4),
        }
    )
    out.attrs["path"] = str(path)
    return out


def compare_with_sheet(inv: pd.DataFrame, sheet: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Disagreements between the spreadsheet and the files actually on disk."""
    cols = ["subject", "session", "issue"]
    if sheet is None:
        return pd.DataFrame(columns=cols)
    rows: List[Dict[str, Any]] = []
    for r in inv.itertuples():
        s = sheet[sheet["subject"] == r.subject]
        if s.empty:
            rows.append(dict(subject=r.subject, session=r.session, issue="not listed in spreadsheet"))
            continue
        for mod, on_disk in (("eeg", r.has_eeg), ("nirs", r.has_nirs)):
            listed = bool(s.iloc[0][f"{r.session}_{mod}"])
            if listed != on_disk:
                rows.append(
                    dict(
                        subject=r.subject,
                        session=r.session,
                        issue=f"{mod}: spreadsheet={'seen' if listed else 'blank'}, "
                        f"disk={'present' if on_disk else 'missing'}",
                    )
                )
    return pd.DataFrame(rows, columns=cols)
