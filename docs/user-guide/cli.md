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

Flux grid options (optional):
- `--fluxgrid-min`, `--fluxgrid-max`, `--fluxgrid-n` configure the shared FluxGrid used for flux marginalization. If unset, defaults are 1e-18, 1e-14, 128. When `--F50` or `--f-lim` are provided, the grid is automatically expanded to comfortably straddle the threshold.

Example:
- `jlc classify input.csv --out output.csv --F50 1.5e-17 --w 5e-18`

Posterior control (diagnostics/ablation):
- `--evidence-only`: disable per-row rate priors and use evidences only for posteriors.
- `--no-global-priors`: ignore any global prior weights (e.g., PPP expected counts) during normalization.

Outputs include per-label columns like `logZ_lae`, `logZ_oii`, `logZ_fake`, `p_lae`, `p_oii`, `p_fake`, plus diagnostics described below.

Note on terminology: The `logZ_<label>` columns now derive from each label’s `extra_log_likelihood(row, ctx)` under the refactored architecture (flux-marginalized measurement evidence). The legacy `log_evidence` name remains as a deprecated shim in label classes and will be removed after a transition period.

## jlc simulate

Generate a mock catalog over a rectangular sky region and classify it with the built-in label models.

Common options:
- `--ra-low/--ra-high` and `--dec-low/--dec-high`: sky bounds (deg)
- `--wave-min/--wave-max`: wavelength band (Å)
- `--f-lim`: selection flux threshold guiding completeness and priors (used if smooth tanh not set)
- `--F50`, `--w`: enable smooth tanh completeness C(F) = 0.5(1 + tanh((F−F50)/w)); if set, they take precedence over `--f-lim`.
- `--flux-err`: measurement noise (stddev) applied when generating measurements
- `--snr-min`: minimum `flux_hat/flux_err` required for a detection to appear in the simulated catalog (applied after measurement noise)
- `--out-catalog`: path to save the simulated input catalog
- `--out-classified`: path to save the classification results
- `--plot-prefix`: if provided, write `PREFIX_wave.png` and `PREFIX_flux.png`
- Flux grid options (shared by simulator and engine): `--fluxgrid-min`, `--fluxgrid-max`, `--fluxgrid-n`. Defaults are 1e-18, 1e-14, 128, and the grid auto-expands to cover `--F50`/`--f-lim` thresholds.

Additional simulator diagnostics:
- `--ppp-parity-check`: run both the engine-aligned simulator and the legacy PPP simulator in a dry-run mode and log expected counts/volumes from each for comparison. This does not change outputs; it only logs parity information to help validate the refactor.
- `--parity-report PATH`: if provided with `--ppp-parity-check`, write a machine-readable JSON summary to PATH (same content as `ctx.config['ppp_parity_summary']`). See the Refactor Architecture page for a worked example and a sample JSON snippet.

Two simulation modes are available:

1) Simple fraction-based simulator (default)
- `--n`: number of sources to generate
- `--lae-frac`, `--oii-frac`, `--fake-frac`: class fractions (sum ≈ 1)

2) Model-driven PPP simulator
- `--from-model`: enable PPP mode
- `--fake-rate`: fake rate density per sr per Å (used in simulator and as Fake rate prior)
- `--calibrate-fake-rate-from CSV`: estimate a homogeneous fake rate ρ̂ from a CSV of virtual detections using the current sky box and wavelength band; overrides `--fake-rate` when successful
- `--validate-fake-rate-from CSV`: validate the provided or calibrated fake rate ρ by comparing observed counts in the CSV to the expectation N̂=ρ·Ω·Δλ; logs a summary and stores a machine-readable result in `ctx.config['fake_rate_validation']`
- `--nz`: redshift grid resolution (default 256)
- `--volume-mode {real,virtual}`: virtual suppresses physical labels (fake-only)

Note on legacy simulator:
- The old legacy PPP simulator pathway has been removed. The engine-aligned `simulate_field` path is the sole backend. Passing `--legacy-ppp` will now produce an error with guidance.

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

## Luminosity function outputs

You can optionally generate binned luminosity functions (LFs) for LAE and OII from the simulated run:

- `--out-lf-observed CSV`: uses the simulated ground-truth `true_class` to make a binned, observed LF per label.
- `--out-lf-inferred CSV`: uses the classifier’s posteriors `p_lae`/`p_oii` to make a binned, inferred LF per label.
- `--lf-bins N`: number of logarithmic (dex) luminosity bins (built around each label’s L* by default).
- `--lf-nz N`: redshift grid size used to compute LF volumes Ω·∫(dV/dz)dz; increase for higher accuracy (default 2048).
- `--lf-plot-prefix PREFIX`: if provided, LF plots are saved as `PREFIX_lae.png` and `PREFIX_oii.png`. These plots now include:
  - binned LF with error bars (observed or inferred, whichever was computed),
  - the default Schechter model curve from the registry’s lae_lf/oii_lf,
  - lightly transparent inferred per-object scatter points (posterior-weighted) for visualizing distribution.

Notes:
- Luminosities are computed from measured flux via L = 4π (d_L[Mpc]·MPC_TO_CM)^2 F with d_L from the configured cosmology and z inferred from `wave_obs` and the label’s rest wavelength.
- Volumes use the label-specific comoving volume Ω·∫(dV/dz)dz across the wavelength band. In PPP mode, these volumes are taken directly from the simulator for exact consistency; otherwise they are computed with the specified `--lf-nz` and matched to PPP behavior (including a low‑z clamp). The simulate command prints the LF volumes used and warns if they differ from PPP volumes by >1%.

## Outputs and diagnostics

Classification outputs include:
- `logZ_<label>`: measurement-only evidence (flux-marginalized)
- `p_<label>`: posterior after combining logZ with log(rate) and optional global priors
- `rate_<label>`: observed-space rate density for each label at the row’s wavelength
- `rate_phys_total`, `rate_fake_total`, `prior_odds_phys_over_fake`
- `log_prior_weight_<label>`: if global prior weights are applied (e.g., PPP expected counts), the per-label log weight used (same across rows) is emitted for auditability.
- If Fake mixture is enabled, per-component columns like `rate_fake_sky_residual`, `rate_fake_noise` summing to `rate_fake_total`.
