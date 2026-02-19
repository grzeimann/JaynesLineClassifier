# Installation

We recommend using Conda (via conda-forge) to manage scientific dependencies, and then installing JaynesLineClassifier (JLC) into that environment with pip in editable mode.

## Create the Conda environment

- conda env create -f environment.yml
- conda activate jlc

Update an existing environment after changes:
- conda env update -f environment.yml --prune

Notes
- Binary dependencies (numpy, pandas, astropy, scipy, matplotlib) are managed by conda-forge via environment.yml.
- Python 3.9+ is supported; environment.yml pins a version tested by the project.

## Install the package into the active env

From the repository root:
- python -m pip install -e .

This installs the console script `jlc` and the `jlc` Python package in editable/development mode.
