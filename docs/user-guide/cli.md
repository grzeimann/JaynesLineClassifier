# CLI Usage

The `jlc` CLI provides two primary commands: `classify` and `simulate`.

## jlc classify

Classify a catalog CSV and write an output CSV with evidences, posterior probabilities, and rate diagnostics.
Required columns in input CSV:
- `wave_obs` (Å), `flux_hat` (flux units), `flux_err` (same units as flux_hat)
- Optional: `ra`, `dec` for reference

Selection model options (optional):
- `--F50` and `--w` enable a smooth tanh completeness C(F) = 0.5(1 + tanh((F−F50)/w)). If set, they take precedence over `--f-lim`.
- `--F50-table` and `--w-table` load wavelength-binned tables for F50(λ) and w(λ) from `.npz` or `.csv` files. Tables override scalar values within their wavelength domains.

Example:
- `jlc classify input.csv --out output.csv --F50 1.5e-17 --w 5e-18`

Outputs include per-label columns like `logZ_lae`, `logZ_oii`, `logZ_fake`, `p_lae`, `p_oii`, `p_fake`, plus diagnostics described below.

## jlc simulate

Generate a mock catalog over a rectangular sky region and classify it with the built-in label models.

Common options:
- `--ra-low/--ra-high` and `--dec-low/--dec-high`: sky bounds (deg)
- `--wave-min/--wave-max`: wavelength band (Å)
- `--f-lim`: selection flux threshold guiding completeness and priors (used if smooth tanh not set)
- `--F50`, `--w`: enable smooth tanh completeness C(F) = 0.5(1 + tanh((F−F50)/w)); if set, they take precedence over `--f-lim`.
- `--flux-err`: measurement noise (stddev) applied when generating measurements
- `--out-catalog`: path to save the simulated input catalog
- `--out-classified`: path to save the classification results
- `--plot-prefix`: if provided, write `PREFIX_wave.png` and `PREFIX_flux.png`

Two simulation modes are available:

1) Simple fraction-based simulator (default)
- `--n`: number of sources to generate
- `--lae-frac`, `--oii-frac`, `--fake-frac`: class fractions (sum ≈ 1)

2) Model-driven PPP simulator
- `--from-model`: enable PPP mode
- `--fake-rate`: fake rate density per sr per Å (used in simulator and as Fake rate prior)
- `--nz`: redshift grid resolution (default 256)
- `--volume-mode {real,virtual}`: virtual suppresses physical labels (fake-only)

Fake λ-PDF calibration and cache I/O:
- `--fake-lambda-calib CSV`: build an empirical wavelength PDF from `wave_obs` in the CSV (e.g., from virtual detections)
- `--fake-lambda-nbins N`: number of bins for λ-PDF (default 200)
- `--fake-lambda-cache-in PATH`: load a precomputed λ-PDF cache (.npz)
- `--fake-lambda-cache-out PATH`: save a built λ-PDF cache (.npz)

Effective search measure knobs (apply to all label rates):
- `--n-fibers`, `--ifu-count`, `--exposure-scale`, `--search-measure-scale`

Example (PPP mode):

```bash
jlc simulate --from-model \
  --ra-low 150 --ra-high 150.5 --dec-low 0 --dec-high 0.5 \
  --wave-min 3500 --wave-max 5500 \
  --f-lim 2e-17 --flux-err 1e-17 \
  --fake-rate 2e3 --nz 256 \
  --out-catalog sim_catalog.csv --out-classified sim_classified.csv \
  --plot-prefix sim
```

## Outputs and diagnostics

Classification outputs include:
- `logZ_<label>`: measurement-only evidence (flux-marginalized)
- `p_<label>`: posterior after combining logZ with log(rate) and optional global priors
- `rate_<label>`: observed-space rate density for each label at the row’s wavelength
- `rate_phys_total`, `rate_fake_total`, `prior_odds_phys_over_fake`
- If Fake mixture is enabled, per-component columns like `rate_fake_sky_residual`, `rate_fake_noise` summing to `rate_fake_total`.
