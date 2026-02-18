# JaynesLineClassifier

A Jaynesian inference engine for labeling emission line sources by combining spectral data, deep photometry, and luminosity-function priors.

## Install (Conda only)

The recommended way to set up a working environment is with Conda using the conda-forge channel. Use the provided environment.yml to create the environment, then install this project into that environment.

1) Create and activate the Conda environment from environment.yml
- macOS/Linux/Windows (PowerShell or cmd):
  - conda env create -f environment.yml
  - conda activate jlc

2) Install JaynesLineClassifier into the active Conda environment
- From the repository root:
  - python -m pip install -e .

Notes
- We use pip only to install this repository into the already-activated Conda environment; all binary dependencies (numpy, pandas, astropy, matplotlib) come from conda-forge via environment.yml.
- To update an existing environment after changes to environment.yml:
  - conda env update -f environment.yml --prune
- Python 3.9+ is supported; 3.10 is pinned in the environment.yml for reproducibility.

## Verify the installation
- python -c "import jlc, sys; print('jlc OK', sys.version)"
- jlc --help

## Quickstart
Given a CSV file with at least the columns: id, wave_obs, flux_hat, flux_err

- jlc classify path/to/input.csv --out path/to/output.csv

The output CSV will include, for each configured label (default: lae, oii, fake):
- logZ_<label>
- p_<label>

The p_<label> values per row should sum to approximately 1.

## Uninstall / Update
- To update from the repo in editable mode, run again from repo root:
  - python -m pip install -e .
- To remove the environment entirely:
  - conda remove -n jlc --all
