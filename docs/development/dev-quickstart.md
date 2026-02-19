# Developer Quickstart

This page is for contributors who want to run the project and build the docs locally.

## 1) Create the environment and install the package

```bash
conda env create -f environment.yml
conda activate jlc
python -m pip install -e .
```

## 2) Run the CLI locally

- Classify an input catalog:
  ```bash
  jlc classify path/to/input.csv --out out.csv
  ```
- Simulate and classify (PPP mode):
  ```bash
  jlc simulate --from-model \
    --ra-low 150 --ra-high 150.5 --dec-low 0 --dec-high 0.5 \
    --wave-min 3500 --wave-max 5500 \
    --f-lim 2e-17 --flux-err 1e-17 \
    --fake-rate 2e3 --nz 256 \
    --out-catalog sim_catalog.csv --out-classified sim_classified.csv
  ```

## 3) Build the documentation (MkDocs)

MkDocs is not part of the Conda environment by default. Install it into your active env:

```bash
python -m pip install -U mkdocs
```

From the repository root, serve the docs locally:

```bash
mkdocs serve
```

Then open the URL printed by mkdocs (usually http://127.0.0.1:8000) to preview the site. To build static HTML instead:

```bash
mkdocs build
```

## 4) Project layout (high level)

- `src/jlc/cli/main.py`: CLI entry points and argument parsing
- `src/jlc/engine/engine.py`: posterior computation from evidences and rate priors
- `src/jlc/labels/`: label models for LAE, OII, and Fake
- `src/jlc/simulate/`: simple and PPP simulators
- `src/jlc/rates/observed_space.py`: observed-space rate utilities (effective_search_measure, fake λ-PDF helpers)
- `src/jlc/population/schechter.py`: Schechter luminosity function
- `src/jlc/cosmology/lookup.py`: Astropy-backed cosmology

## 5) Contributing

See Community → Contributing for guidelines. Open issues/PRs are welcome.
