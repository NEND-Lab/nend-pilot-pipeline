"""Configuration for the NEND pilot analysis pipeline.

Everything that can change between runs lives here, so a run is fully described by
one :class:`Config` object which is written verbatim into the run log.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Dict, List, Optional, Tuple

__all__ = ["Config", "default_config"]

# Environment variable holding the path to the BIDS-style study folder.
BIDS_ROOT_ENV = "NEND_BIDS_ROOT"


@dataclass
class Config:
    """Parameters for a complete analysis run.

    Attributes:
        bids_root: Root of the study folder (``sub-<id>/ses-<cond>/...``).
            Read from ``NEND_BIDS_ROOT`` when not given explicitly. No default is
            hard-coded, so a fresh clone cannot accidentally point at a private
            dataset share.
        subjects: Participant labels to analyse; ``None`` means every participant
            discovered under ``bids_root``.
        condition_a: Label of the first condition (the "A" side of every contrast).
        condition_b: Label of the second condition.
        marker_is_trial_start: ``True`` if an automarker marks the *start* of a
            5-s task item, ``False`` if it marks the *end*. Verify against the
            task protocol; the pipeline cannot infer this from the data.
        min_trials: Minimum number of usable trials for a session to be summarised.
        marker_regex: Only keep markers whose code matches this regular expression.
        events_fallback_to_csv: Use the EEG file's ``event_marker`` column when the
            automarker file is missing or empty.
    """

    # ---- where things are -----------------------------------------------------------------------
    bids_root: str = field(default_factory=lambda: os.environ.get(BIDS_ROOT_ENV, ""))
    availability_xlsx: Optional[str] = None
    output_dir: Optional[str] = None

    # ---- which data -----------------------------------------------------------------------------
    subjects: Optional[List[str]] = None
    condition_a: str = "fasting"
    condition_b: str = "nonfasting"

    # ---- protocol -------------------------------------------------------------------------------
    baseline_s: float = 30.0
    trial_s: float = 5.0
    marker_is_trial_start: bool = True
    min_trials: int = 5
    marker_regex: Optional[str] = None
    events_fallback_to_csv: bool = True

    # ---- EEG ------------------------------------------------------------------------------------
    eeg_sfreq: float = 256.0
    eeg_channels: Dict[str, str] = field(
        default_factory=lambda: {"TP9": "TP9_uV", "AF7": "AF7_uV", "AF8": "AF8_uV", "TP10": "TP10_uV"}
    )
    frontal_channels: Tuple[str, str] = ("AF7", "AF8")
    eeg_band: Tuple[float, float] = (1.0, 40.0)
    eeg_reference: Optional[str] = None
    eeg_reject_uv: float = 200.0
    eeg_flat_uv: float = 0.5
    eeg_discard_start_s: float = 5.0
    eeg_baseline: Optional[Tuple[float, float]] = None
    eeg_min_baseline_windows: int = 5
    eeg_min_trial_windows: int = 2
    welch_win_s: float = 2.0
    bands: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: {
            "delta": (1.0, 4.0),
            "theta": (4.0, 8.0),
            "alpha": (8.0, 13.0),
            "beta": (13.0, 30.0),
            "gamma": (30.0, 40.0),
        }
    )

    # ---- fNIRS ----------------------------------------------------------------------------------
    nirs_unit_scale: float = 1e6
    nirs_discard_start_s: float = 3.0
    nirs_band: Tuple[Optional[float], Optional[float]] = (None, 0.2)
    nirs_tddr: bool = True
    nirs_motion_z: float = 3.5
    nirs_max_motion_frac: float = 0.30
    nirs_shift_s: float = 4.0
    nirs_use_labels: bool = True
    nirs_min_baseline_s: float = 3.0
    nirs_block_s: float = 50.0

    # ---- statistics -----------------------------------------------------------------------------
    primary: Tuple[str, ...] = ("theta_front_db", "alpha_front_db", "faa_alpha", "hbo_pfc", "hbr_pfc")
    n_perm: int = 2000
    n_boot: int = 5000
    alpha: float = 0.05
    min_pairs_inference: int = 3
    min_pairs_cluster: int = 6
    seed: int = 42

    # ---- presentation ---------------------------------------------------------------------------
    max_inline_subjects: int = 2

    def resolved_bids_root(self) -> str:
        """Return the study root, raising a helpful error when it has not been set."""
        if not self.bids_root:
            raise ValueError(
                "No study folder set. Pass --bids-root, or set the "
                f"{BIDS_ROOT_ENV} environment variable (for example "
                "$env:NEND_BIDS_ROOT = 'D:/NEND-pilot')."
            )
        return self.bids_root

    def to_dict(self) -> Dict[str, Any]:
        """Plain-data view of the configuration, for the run log."""
        return asdict(self)

    def replace(self, **changes: Any) -> "Config":
        """Return a copy with the given fields replaced."""
        return replace(self, **changes)


def default_config(**overrides: Any) -> Config:
    """Build a configuration, applying any overrides."""
    return Config(**overrides)
