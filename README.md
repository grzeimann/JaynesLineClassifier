# JaynesLineClassifier

A Jaynesian inference engine for labeling emission-line candidates by combining spectral measurements, selection models, cosmology + luminosity-function priors, and observed-space rate densities.

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
- We use pip only to install this repository into the already-activated Conda environment; all binary dependencies (numpy, pandas, astropy, scipy, matplotlib) come from conda-forge via environment.yml.
- To update an existing environment after changes to environment.yml:
  - conda env update -f environment.yml --prune
- Python 3.9+ is supported; 3.10 is pinned in the environment.yml for reproducibility.

## What’s new (Phase 2)
Phase 2 extends the framework with a virtual-volume workflow and empirical calibration tools for the fake rate model, plus richer control over the effective search measure used in rate priors.

Highlights:
- Virtual volume mode (--volume-mode virtual): suppresses physical sources while keeping the fake model active, enabling calibration from negative/blank regions.
- Empirical fake λ-intensity: build a wavelength PDF from virtual detections and modulate the fake rate prior by its shape.
- Cache I/O for fake λ-PDF: save/load calibrated shape for reuse across runs.
- Effective search measure knobs: configure simple multiplicative factors (n_fibers, ifu_count, exposure_scale, search_measure_scale) applied consistently to all label rates.

## What’s new (Phase 1)
Phase 1 introduces a unified observed-space rate prior and integrates it with the existing data evidence for each label. This brings the CLI, simulator, and outputs into a consistent probabilistic framework.

Highlights:
- Each label now provides a rate_density(row, ctx) in a common observed measure dλ · dF · dA_fiber.
- Physical labels (LAE, OII) compute rates from LF × dV/dz × selection, with correct luminosity (Mpc→cm) handling and Jacobians.
- The Fake label has a simple, configurable rate per steradian per Ångström.
- Posterior probabilities are formed from logZ (data evidence) plus log(rate) and optional global prior weights.
- Output CSVs now include per-row diagnostics (rate_* and totals).

## Quickstart: Classify a catalog
Given a CSV file with at least the columns: id, wave_obs, flux_hat, flux_err

- jlc classify path/to/input.csv --out path/to/output.csv

The output CSV will include, for each configured label (default: lae, oii, fake):
- logZ_<label>: flux-marginalized log-evidence from the measurement model(s)
- p_<label>: posterior probability after combining evidence with rate priors

New diagnostic columns (per row):
- rate_<label>: observed-space rate density used for that label (per sr per Å)
- rate_phys_total: sum of physical label rates
- rate_fake_total: fake rate (present if the fake label is in the registry)
- prior_odds_phys_over_fake: rate_phys_total / rate_fake_total (nan if fake rate is zero)
- If the Fake mixture is enabled (default), additional per-component columns are emitted, e.g.,
  rate_fake_sky_residual and rate_fake_noise, which sum to rate_fake_total.

How posteriors are computed:
- For each label y, logP_y ∝ logZ_y + log r_y + w_y, where r_y is the per-row rate_density and w_y is an optional global prior weight (see PPP mode below). Probabilities are then normalized across labels.

## Simulation mode
You can generate mock catalogs over a rectangular sky box and classify them with the built‑in labels (LAE, OII, Fake). Two modes are available:

1) Simple fraction-based simulator (original)
- Uniform sky within RA/DEC limits.
- User-provided class fractions control label ratios.
- Observed wavelengths are uniform within [wave_min, wave_max].
- Fluxes per class are drawn from simple placeholders.
- A hard selection is applied on measured flux (flux_hat > f_lim).

Example:
- jlc simulate --n 2000 --ra-low 150 --ra-high 160 --dec-low -2 --dec-high 2 \
  --wave-min 4800 --wave-max 9800 --f-lim 1e-17 --flux-err 5e-18 \
  --lae-frac 0.3 --oii-frac 0.3 --fake-frac 0.4 \
  --out-catalog sim_catalog.csv --out-classified sim_classified.csv \
  --plot-prefix sim

2) Model-driven PPP simulator (Phase 1)
- Uses a Poisson point process whose intensity is determined by the label models:
  - LAE/OII counts follow the luminosity function (Schechter) integrated over flux and the comoving volume element dV/dz within the wavelength band, scaled by the sky solid angle.
  - Selection is incorporated via the selection completeness S(F, λ_obs) when building the intensity.
  - Observed wavelengths are distributed according to the induced redshift prior (dV/dz × |dz/dλ|) for each label.
  - Fluxes are drawn from the LF transformed into flux-space at the sampled redshift.
  - Fakes are generated by a homogeneous PPP with a configurable rate per steradian per Ångström.
- No additional hard selection is applied after measurement; selection is already encoded in the PPP intensity.
- During simulation, expected counts per label and label-specific comoving volumes are printed for diagnostics and also stored in the context.

Example (PPP mode):
- jlc simulate --from-model --ra-low 150 --ra-high 160 --dec-low -2 --dec-high 2 \
  --wave-min 4800 --wave-max 9800 --f-lim 1e-17 --flux-err 5e-18 \
  --fake-rate 2e3 \
  --out-catalog sim_catalog.csv --out-classified sim_classified.csv \
  --plot-prefix sim

Key options for PPP mode:
- --from-model: enable PPP simulator.
- --fake-rate: fake rate density per steradian per Ångström (default 0.0). This also seeds the Fake label’s rate prior via ctx.config["fake_rate_per_sr_per_A"].
- --nz: number of redshift grid points used to construct the intensity (default 256).

Columns in the simulated catalog (both modes):
- ra, dec, true_class, wave_obs, flux_hat, flux_err, snr (if --snr-min provided, only rows with snr >= threshold are kept)

Notes:
- The selection model used during classification is configured with the same --f-lim value. In PPP mode, the selection also shapes the generated population via S(F, λ_obs).
- The PPP simulator relies on the default cosmology (Astropy Planck18) and the placeholder Schechter LF parameters configured in the CLI builder.
- In PPP mode, the classifier will (by default) use the simulator’s expected counts as global prior weights when normalizing posteriors, ensuring class imbalance is reflected.

## Plots
If you pass --plot-prefix NAME to the simulate command, two figures are written:
- NAME_wave.png: histogram of observed wavelengths by true_class.
- NAME_flux.png: histogram of measured fluxes by true_class. The x-axis is logarithmic with a lower bound at 1e-18, and the y-axis uses a logarithmic scale for counts.

## Configuration and context keys
These keys are set by the CLI and used internally by models:
- f_lim: flux threshold guiding selection completeness and the Fake flux prior.
- F50, w: optional smooth tanh selection parameters. If both are provided, completeness uses C(F) = 0.5(1 + tanh((F−F50)/w)) and they take precedence over f_lim.
- wave_min, wave_max: wavelength band for the run.
- fake_rate_per_sr_per_A: PPP-mode fake rate density; also used by the Fake label’s rate prior.
- ppp_expected_counts: dict of expected counts by label computed during PPP simulation (used as global prior weights by the classifier).
- ppp_label_volumes: dict of per-label comoving volumes (Mpc^3) probed by the requested sky + wavelength region.

## Under the hood (for developers)
- Observed-space rate density r(λ) for LAE/OII: ∫ dF [ dV/dz × φ(L(F,z)) × dL/dF × S(F,λ) × |dz/dλ| ] with d_L in Mpc converted to cm for luminosity.
- Posteriors are computed by combining logZ (from measurement likelihoods marginalized over latent flux) with log(rate) and optional global weights, then normalized across labels.
- effective_search_measure(row, ctx) multiplies all rates; currently returns 1.0 (configurable) and serves as the hook for Phase 2 virtual-volume work.
