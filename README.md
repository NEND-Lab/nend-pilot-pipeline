# NEND Pilot Analysis Pipeline (development in progress)

Analysis code for the **NEND Pilot Study**, a within-participant crossover feasibility study of fasting, cognitive tasks, and prefrontal activity. The study uses portable **Muse EEG** and **Mendi fNIRS** recordings collected during fasting and non-fasting sessions.

This is a **pilot study**: its purpose is to assess recording feasibility and data quality, and to estimate possible effects for future research—not to provide clinical conclusions.

## Study at a glance

Each participant is intended to complete two sessions: one after an overnight fast and one after eating. Participants perform short working-memory and arithmetic tasks while the devices record complementary signals:

- **Muse EEG** records electrical activity from four sites: TP9, AF7, AF8, and TP10.
- **Mendi fNIRS** records changes in oxygenated haemoglobin (HbO) and deoxygenated haemoglobin (HbR) over the left and right forehead.

Comparing each participant with themselves reduces the influence of differences between people. The study protocol describes a randomised, counterbalanced session order.

**Study conditions described in the protocol:** fasting means at least 12 hours without caloric intake, with water permitted; the non-fasting session follows a meal. Consult the approved study protocol and acquisition records for the conditions actually used for each recording.

## What the pipeline does

The pipeline discovers recordings in a BIDS-style folder and produces analyses and quality-control outputs for the available data. These include:

- An inventory of EEG and fNIRS files, optionally checked against an `Available datasets.xlsx` spreadsheet
- An audit of EEG event markers
- EEG task-related features, including frontal theta and alpha power relative to rest and frontal alpha asymmetry
- fNIRS HbO and HbR summaries and quality checks
- Within-participant fasting versus non-fasting comparisons, where paired data are available
- Exploratory EEG–fNIRS coupling summaries
- Tables, figures, and a run log

**Scope note:** The research protocol also proposes behavioural outcomes (such as arithmetic accuracy and reaction time) and future event-related potential (ERP) analyses. Behavioural outcomes require a separate script, and the protocol describes ERP processing as a *planned addition*. Do not interpret this repository's outputs as including those analyses unless they have been implemented and validated.

## Installation

Requirements: Python 3.10 or newer and access to the study recordings.

```bash
git clone https://github.com/NEND-Lab/nend-pilot-pipeline.git
cd nend-pilot-pipeline
python -m pip install -e .
```

For development and tests:

```bash
python -m pip install -e ".[dev]"
```

## Input data

Point the pipeline at a study folder containing `sub-*` participant directories and `ses-*` session directories. A simplified layout is:

```text
study-folder/
├── sub-NEND001/
│   ├── ses-fasting/
│   │   ├── eeg/
│   │   │   ├── ..._eeg.csv
│   │   │   └── ..._events.tsv
│   │   └── nirs/
│   │       └── ..._nirs.json
│   └── ses-nonfasting/
│       ├── eeg/
│       └── nirs/
└── sub-NEND002/
    └── ...
```

The availability spreadsheet is optional. The pipeline can run without it; when present, it reports discrepancies between the spreadsheet and files found on disk.

**Timing matters:** Muse and Mendi use separate clocks. EEG events are aligned to EEG markers; fNIRS data are aligned to labels in the fNIRS recording. Do not assume that a timestamp from one device can be applied directly to the other. The protocol also notes that marker meaning and timing need confirmation against the acquisition procedure before interpreting event-locked results.

## Run the pipeline

Analyse participants found in the study folder:

```bash
nend-analyse --bids-root /path/to/study-folder
```

Select participants and an output directory:

```bash
nend-analyse \
  --bids-root /path/to/study-folder \
  --subjects NEND001 NEND002 \
  --out /path/to/results
```

You can also set the `NEND_BIDS_ROOT` environment variable instead of passing `--bids-root`. For all available settings and their current defaults, run:

```bash
nend-analyse --help
```

Before analysing event-locked EEG measures, check whether each marker denotes the **start or end** of a task item. Use `--marker-is-trial-end` only when the acquisition records establish that markers denote trial ends.

## Results

By default, results are written to `derivatives/nend_pipeline` under the study folder; use `--out` to change this. Outputs may include:

| Output | Purpose |
| --- | --- |
| `marker_audit.csv` | EEG marker checks |
| `nirs_quality.csv` | fNIRS quality information |
| `session_table.csv` | Participant/session summaries |
| `group_comparison.csv` | Across-participant comparisons |
| `trial_level_comparison.csv` | Trial-level exploratory comparisons |
| `eeg_trials.csv` and `nirs_bins.csv` | Modality-specific results |
| `coupling.csv` | EEG–fNIRS coupling results |
| `spreadsheet_mismatch.csv` | Discrepancies with the optional availability spreadsheet |
| `figures/` | Generated plots |
| `run_log.json` | Configuration, software information, and analysed or skipped sessions |

Files for empty results may not be created. Review quality and marker-audit outputs **before** interpreting comparisons.

## Interpretation and limitations

- The sample is a feasibility pilot; effect sizes and uncertainty are more informative than a significant/non-significant label alone.
- Four dry EEG electrodes limit spatial detail and are susceptible to movement and blink artefacts.
- The fNIRS device has no short-separation channels to directly remove scalp blood-flow effects. HbO and HbR should be considered together.
- Separate device clocks and uncertain marker timing limit precise cross-device, single-trial interpretation.
- Missing or low-quality sessions must be reported rather than silently treated as complete pairs.

See the research protocol for the study rationale, acquisition procedures, planned hypotheses, and full analysis plan. **The protocol describes intended procedures; check the code, run log, and acquisition records to establish what was actually done for a particular analysis.**

## Tests

Tests use synthetic recordings and do not require participant data:

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

## Data privacy and governance

Do not commit participant recordings, identifying information, consent records, credentials, or generated results to this public repository. Keep the participant identification key separate from coded research data. Share study data only under the applicable consent, ethics approval, and institutional governance arrangements.

The supplied protocol identifies ethics approval and study registration as **in progress** in its administrative information. Do not present either as complete until its status has been confirmed and this README has been updated.

## License

This repository is licensed under the [MIT License](LICENSE).
