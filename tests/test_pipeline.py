"""Unit and end-to-end tests for the NEND pipeline.

Run with ``pytest`` from the repository root.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from nend_pipeline import Config, run_analysis, save_results          # noqa: E402
from nend_pipeline.cluster import cluster_perm_paired                 # noqa: E402
from nend_pipeline.inventory import norm_session, norm_subject      # noqa: E402
from nend_pipeline.loaders import trial_windows                      # noqa: E402
from nend_pipeline.stats import bh_fdr, paired_summary               # noqa: E402
from synthetic import build_dataset                                  # noqa: E402


@pytest.fixture(scope="module")
def study(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("nend")
    return build_dataset(tmp / "NEND-pilot", n_subjects=4)


@pytest.fixture(scope="module")
def results(study):
    cfg = Config(bids_root=str(study), subjects=None, n_perm=60, n_boot=200)
    return run_analysis(cfg, log=lambda *a, **k: None)


# ---------------------------------------------------------------- naming and layout
def test_norm_subject():
    assert norm_subject("NEND002") == "sub-NEND002"
    assert norm_subject("sub-NEND002") == "sub-NEND002"


@pytest.mark.parametrize(
    ("folder", "expected"),
    [("ses-fasting", "fasting"), ("ses-nonfasting", "nonfasting"), ("ses-non-fasting", "nonfasting")],
)
def test_norm_session(folder, expected):
    assert norm_session(folder) == expected


def test_bids_root_must_be_set():
    with pytest.raises(ValueError, match="NEND_BIDS_ROOT"):
        Config(bids_root="").resolved_bids_root()


def test_missing_study_folder_raises():
    with pytest.raises(FileNotFoundError):
        run_analysis(Config(bids_root="/definitely/not/here"), log=lambda *a, **k: None)


def test_trial_windows_marker_semantics():
    cfg = Config()
    markers = pd.DataFrame({"onset": [10.0, 15.0], "code": ["a", "a"], "trial": [1, 2]})
    start = trial_windows(markers, cfg)["start"].to_numpy()
    assert list(start) == [10.0, 15.0]
    end = trial_windows(markers, cfg.replace(marker_is_trial_start=False))["start"].to_numpy()
    assert list(end) == [5.0, 10.0]


# ---------------------------------------------------------------- statistics
def test_bh_fdr_monotonic():
    p = np.array([0.01, 0.02, 0.5, np.nan])
    q = bh_fdr(p)
    assert q[0] < q[1] < q[2]
    assert np.isnan(q[3])


def test_paired_summary_needs_minimum_pairs():
    cfg = Config(min_pairs_inference=3)
    out = paired_summary(np.array([1.0]), np.array([0.0]), cfg)
    assert out["n_pairs"] == 1
    assert np.isnan(out["p_wilcoxon"])


def test_paired_summary_detects_consistent_difference():
    cfg = Config(min_pairs_inference=3, n_boot=200)
    a = np.array([2.0, 3.0, 4.0, 5.0])
    b = np.array([1.0, 2.0, 3.0, 4.0])
    out = paired_summary(a, b, cfg)
    assert out["n_pairs"] == 4
    assert out["mean_diff"] == pytest.approx(1.0)
    assert out["p_wilcoxon"] < 0.2


def test_cluster_permutation_runs():
    rng = np.random.default_rng(0)
    diff = rng.normal(0, 1, (6, 5)) + np.linspace(0, 3, 5)
    t_map, labels, table = cluster_perm_paired(diff, n_perm=50, alpha=0.05)
    assert t_map.shape == (6, 5)
    assert labels.shape == (6, 5)
    assert len(table) >= 1


# ---------------------------------------------------------------- end to end
def test_analysis_finds_all_sessions(results):
    assert len(results.sessions) == 7            # 4 participants x 2 sessions, minus the missing one
    assert len(results.audit) == 7


def test_audit_flags_empty_marker_file_falls_back_to_csv(results):
    row = results.audit[
        (results.audit["subject"] == "sub-NEND001") & (results.audit["session"] == "fasting")
    ].iloc[0]
    assert row["source"] == "csv"
    assert row["markers"] == 13


def test_audit_reports_marker_spacing(results):
    row = results.audit[results.audit["source"] == "tsv"].iloc[0]
    assert row["median_isi_s"] == pytest.approx(5.0, abs=0.1)
    assert "ISI_IRREGULAR" not in row["flags"]


def test_session_table_has_expected_columns(results):
    t = results.session_table
    for col in ("theta_front_db", "alpha_front_db", "faa_alpha", "hbo_pfc", "hbr_pfc"):
        assert col in t.columns


def test_group_comparison_pairs_present(results):
    g = results.group
    assert not g.empty
    assert g["n_pairs"].max() >= 3
    assert "dz" in g.columns
    assert "p_fdr" in g.columns


def test_nirs_quality_flags_short_baseline(results):
    q = results.nirs_quality
    fasting = q[q["session"] == "fasting"].iloc[0]
    assert "BASELINE" in fasting["flags"]


def test_save_results_writes_expected_files(results, tmp_path):
    from nend_pipeline.export import save_results

    out = save_results(results.config, results, out_dir=tmp_path)
    names = {p.name for p in out.iterdir()}
    for expected in (
        "marker_audit.csv", "nirs_quality.csv", "session_table.csv",
        "group_comparison.csv", "coupling.csv", "run_log.json",
    ):
        assert expected in names
    assert (out / "figures").is_dir()


def test_run_log_contains_config(results, tmp_path):
    import json

    from nend_pipeline.export import save_results

    out = save_results(results.config, results, out_dir=tmp_path)
    log = json.loads((out / "run_log.json").read_text(encoding="utf-8"))
    assert "config" in log
    assert "versions" in log
    assert log["completeness"]["n_sessions"] == len(results.sessions)
