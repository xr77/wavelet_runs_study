# Wavelet Runs Study

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23228383.svg)](https://doi.org/10.5281/zenodo.23228383)

Python tools for extracting spatial wavelet features from fMRI volumes and evaluating category classification across held-out runs.

**Author:** Xueying Ren · University of Pittsburgh

**License:** [MIT](LICENSE) · **Python:** 3.10–3.12

This repository contains source code, documentation, and synthetic tests only. **No laboratory data, participant information, saved study results, or Jupyter notebooks are distributed.** Analysis commands read local inputs and write local outputs; none upload or download data.

## Installation

```bash
git clone https://github.com/xr77/wavelet_runs_study.git
cd wavelet_runs_study
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
```

For development, install `python -m pip install -e ".[dev]"`. Optional XGBoost support is available with `python -m pip install -e ".[xgboost]"`; the default classifier is Gaussian naive Bayes. See the [analysis guide](docs/analysis.md) for preprocessing and model choices.

## Quick start: synthetic data

```bash
wavelet-runs demo --output outputs/demo_features.npz
python examples/synthetic_workflow.py
```

The first command generates four seeded synthetic volumes and a feature array with shape `(4, 5, 28)`: time points, wavelet scales, and orientations. The example demonstrates run-wise classification in memory. These are software checks, not scientific findings.

## Analyze your own local data

Supply preprocessed, spatially aligned NIfTI runs in acquisition order and a binary ROI mask:

```bash
wavelet-runs extract \
  --bold /path/to/run_01.nii.gz /path/to/run_02.nii.gz \
  --mask /path/to/roi_mask.nii.gz \
  --zscore --levels 5 \
  --output outputs/features.npz

wavelet-runs labels \
  --input /path/to/conditions.mat --key conds_short_tlrc \
  --shift 2 --runs-from outputs/features.npz \
  --output outputs/labels.npy

wavelet-runs classify \
  --features outputs/features.npz --labels outputs/labels.npy \
  --classifier gaussian-nb --output outputs/classification.json
```

The condition matrix is binary with axes `(condition, time)`; label zero denotes rest. This example shifts labels within each run. Choose delay and boundary behavior for your experiment. Classification holds out each run, fits a new model per fold and scale, and reports accuracies and held-out confusion matrices. Analyze each subject separately.

Commands refuse to overwrite outputs. Use `wavelet-runs --help` or `wavelet-runs <command> --help` for options. Python users can import functions from `wavelet_runs`.

## Project structure

```text
src/wavelet_runs/
    features.py          # DT-CWT feature calculation
    preprocessing.py     # ROI masking and per-run normalization
    imaging.py           # NIfTI loading and spatial-grid checks
    conditions.py        # Condition labels and explicit time shifts
    classification.py    # Leave-one-run-out evaluation
    storage.py           # Local feature I/O and subject stacking
    cli.py               # Command-line interface
examples/
    synthetic_workflow.py
tests/                  # Tests using generated inputs only
tools/
    check_release.py     # Exclude data, notebooks, and outputs from releases
docs/
    analysis.md          # Inputs, methods, and limitations
    migration.md         # Mapping from original research code
    releasing.md         # Code-only release process
```

## Validation and scope

```bash
pytest
ruff check src tests examples tools
ruff format --check src tests examples tools
python tools/check_release.py
```

The feature statistic is the population variance of the natural logarithm of DT-CWT coefficient magnitudes, calculated over space for each scale and orientation. Undefined statistics remain `NaN` and are rejected by classification. The software does not perform MRI preprocessing, registration, resampling, or automatic statistical inference.

This is a refactoring of exploratory research code, not a claim of exact reproduction of historical results. The original data are not authorized for distribution and are not included. See [migration notes](docs/migration.md) for changes and analyses outside the supported API.

## Citation

> Ren, X. (2026). *Wavelet Runs Study* (Version 0.2.0) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.23228383

Metadata are maintained in [CITATION.cff](CITATION.cff). The [Zenodo archive](https://zenodo.org/records/23228383) contains the same verified source-only ZIP as [GitHub release v0.2.0](https://github.com/xr77/wavelet_runs_study/releases/tag/v0.2.0). The earlier 0.1.0 files remain restricted on Zenodo.

## Contributing

Report issues through [GitHub](https://github.com/xr77/wavelet_runs_study/issues), using synthetic inputs and the exact software version. See [CONTRIBUTING.md](CONTRIBUTING.md). Dependencies retain their own licenses, including the [DT-CWT implementation](https://github.com/rjw57/dtcwt).
