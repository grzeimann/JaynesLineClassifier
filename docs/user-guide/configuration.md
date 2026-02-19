# Configuration

JLC pulls run-time configuration from CLI options and stores them in a shared context passed to models. Important keys:

- `f_lim`: flux threshold guiding selection completeness and Fake priors.
- `wave_min`, `wave_max`: wavelength band for the run (Å).
- `fake_rate_per_sr_per_A`: fake rate density used in PPP mode and by FakeLabel rate priors.
- `ppp_expected_counts`: dict of expected counts by label computed during PPP simulation; used as optional global prior weights when normalizing posteriors.
- `ppp_label_volumes`: dict of per-label comoving volumes (Mpc^3) within the sky box and wavelength window.
- `volume_mode`: `real` (default) or `virtual` (physical labels suppressed).
- `fake_rate_per_sr_per_A`: fake rate density used by the Fake label’s rate prior (can be provided or calibrated).
- `fake_rate_rho_used`: if present, the calibrated/used fake rate ρ (per sr per Å) recorded for traceability; also emitted as a `rho_used` column in outputs.
- Posterior toggles:
  - `use_rate_priors` (default True): if False, disables per-row rate priors (evidence-only posteriors).
  - `use_global_priors` (default True): if False, ignores global prior weights (e.g., PPP expected counts) during normalization.
- Effective search measure knobs (multiply all rates):
  - `n_fibers`, `ifu_count`, `exposure_scale`, `search_measure_scale`.

## Selection model

By default, `SelectionModel` applies a hard-completeness step at `f_lim`:
- completeness(F, λ) = 1 if F > f_lim, else 0.

Alternatively, you can enable a smooth tanh completeness curve with `--F50` and `--w`:
- C(F) = 0.5 · (1 + tanh((F − F50)/w))
- If `F50` and `w` are provided, they take precedence over `f_lim`.

Wavelength-dependent completeness (optional):
- Provide wavelength-binned tables for F50(λ) and/or w(λ) via `--F50-table` and `--w-table`.
- File formats supported: `.npz` (preferred; arrays `bins` and `values`) or `.csv` with two columns: `bin_left,value` per row. The last right edge is reconstructed for CSV using the median bin width.
- Tables override scalar `F50`/`w` within their wavelength domain; outside the table range, scalar values (or legacy behavior) are used.
- Programmatic I/O: `SelectionModel.save_table(path, bins, values)` and `SelectionModel.load_table(path)`.

## Label models

- Physical labels (LAE, OII):
  - `rate_density(row, ctx)`: integrates LF × dV/dz × selection × Jacobians over flux at the candidate’s wavelength.
  - `log_evidence(row, ctx)`: measurement-only evidence marginalized over latent flux with a neutral prior.
- Fake label:
  - `rate_density(row, ctx)`: base fake rate per sr per Å modulated by empirical λ-PDF (if available) and multiplied by effective_search_measure; optionally a simple mixture across components.
  - `log_evidence(row, ctx)`: measurement-only evidence marginalized over latent flux with a neutral prior.

## Caches

- `flux_grid`: a shared quadrature grid (values and log-weights) used to marginalize over latent flux in evidence calculations.
- `fake_lambda_pdf`: optional empirical wavelength PDF cache used to modulate the fake rate prior shape. Built from virtual detections.

## Setting config via CLI

The `simulate` command sets most of these keys directly from its options; see User Guide → CLI for details. When using `classify` directly, defaults are used unless you provide a custom builder.
