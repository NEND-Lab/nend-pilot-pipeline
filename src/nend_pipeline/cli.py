"""Command-line entry point: ``nend-analyse``."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from .config import Config
from .export import save_results
from .pipeline import run_analysis

__all__ = ["main", "build_parser"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nend-analyse",
        description=(
            "Analyse NEND pilot EEG (Muse) and fNIRS (Mendi) recordings. "
            "Reads a BIDS-style study folder and writes tables and figures."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--bids-root",
        default=None,
        help="Study folder containing sub-<id>/ses-<condition>/ folders "
        "(defaults to the NEND_BIDS_ROOT environment variable)",
    )
    parser.add_argument("--out", default=None, help="Where to write results (default: derivatives under the study folder)")
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=None,
        metavar="ID",
        help="Participants to analyse, e.g. NEND002 NEND003 (default: all)",
    )
    parser.add_argument("--condition-a", default="fasting", help="Label of the first condition")
    parser.add_argument("--condition-b", default="nonfasting", help="Label of the second condition")
    parser.add_argument(
        "--marker-is-trial-start",
        dest="marker_is_trial_start",
        action="store_true",
        default=None,
        help="Treat each automarker as the START of its task item (default)",
    )
    parser.add_argument(
        "--marker-is-trial-end",
        dest="marker_is_trial_start",
        action="store_false",
        help="Treat each automarker as the END of its task item",
    )
    parser.add_argument("--baseline-s", type=float, default=30.0, help="Resting baseline length (s)")
    parser.add_argument("--trial-s", type=float, default=5.0, help="Spacing between automarkers (s)")
    parser.add_argument("--reject-uv", type=float, default=200.0, help="Peak-to-peak rejection threshold (uV)")
    parser.add_argument(
        "--reference",
        choices=("none", "average"),
        default="none",
        help="EEG referencing: keep the device reference (default) or average-reference",
    )
    parser.add_argument("--nirs-shift-s", type=float, default=4.0, help="Haemodynamic lag applied to fNIRS bins (s)")
    parser.add_argument("--n-perm", type=int, default=2000, help="Permutations for cluster tests")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-figures", action="store_true", help="Skip figure generation")
    parser.add_argument("--quiet", action="store_true", help="Only print errors")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Run the analysis from the command line.

    Returns:
        Process exit code: 0 on success, 1 on error.
    """
    args = build_parser().parse_args(argv)
    log = (lambda *a, **k: None) if args.quiet else print

    subjects = None
    if args.subjects:
        subjects = args.subjects
        if subjects and len(subjects) == 1 and subjects[0].lower() == "all":
            subjects = None

    cfg = Config(
        bids_root=args.bids_root or "",
        output_dir=args.out,
        subjects=subjects,
        condition_a=args.condition_a,
        condition_b=args.condition_b,
        marker_is_trial_start=(
            True if args.marker_is_trial_start is None else args.marker_is_trial_start
        ),
        baseline_s=args.baseline_s,
        trial_s=args.trial_s,
        eeg_reject_uv=args.reject_uv,
        eeg_reference=None if args.reference == "none" else "average",
        nirs_shift_s=args.nirs_shift_s,
        n_perm=args.n_perm,
        seed=args.seed,
    )

    try:
        results = run_analysis(cfg, log=log)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    out = save_results(cfg, results, log=log)
    log(f"Done in {results.elapsed_s:.1f} s. Output: {out}")

    if not args.no_figures:
        log(f"Figures in {out / 'figures'}")
    return 0


if __name__ == "__main__":       # pragma: no cover
    raise SystemExit(main())
