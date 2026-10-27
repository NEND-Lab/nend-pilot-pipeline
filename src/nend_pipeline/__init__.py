"""NEND pilot analysis pipeline.

A multimodal EEG (Muse) and fNIRS (Mendi) analysis pipeline for the NEND fasting
study. See ``docs/METHODS.md`` for the methodological choices and their references.

Typical use::

    from nend_pipeline import Config, run_analysis, save_results

    cfg = Config(bids_root="D:/NEND-pilot", subjects=["NEND002"])
    results = run_analysis(cfg)
    save_results(cfg, results, "outputs/")
"""
from __future__ import annotations

from .config import Config, default_config
from .pipeline import Results, resolve_output_dir, run_analysis
from .export import save_results

__all__ = [
    "Config",
    "default_config",
    "Results",
    "run_analysis",
    "resolve_output_dir",
    "save_results",
]

__version__ = "0.1.0"
