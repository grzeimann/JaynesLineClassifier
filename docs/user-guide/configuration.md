# Configuration

JLC pulls run-time configuration from CLI options and stores them in a shared context passed to models. Important keys:

> Rename status: The codebase now prefers `extra_log_likelihood` terminology throughout. Engine/CLI/tests use it end‑to‑end; label classes retain a deprecated `log_evidence` shim (DeprecationWarning, stacklevel=2) delegating to `extra_log_likelihood` for one minor version. Custom labels should implement `extra_log_likelihood(row, ctx)` and may keep a thin `log_evidence` shim during the transition window.

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

The SelectionModel now uses S/N-based completeness driven by priors. At runtime, completeness is evaluated via:
- selection.completeness_sn_array(label_name, F_array, wave_true, ...)

How it works:
- A NoiseModel provides sigma(wave_true[, ra, dec, ifu_id]).
- Per-label SNCompletenessModel (e.g., logistic_per_lambda_bin) maps S/N_true → C ∈ [0,1].
- When you load a PriorRecord that contains a `selection` block (default_sigma and selection.sn), the CLI wires these models automatically.
- If no selection prior is provided, completeness defaults to 1.0 (neutral), preserving previous behavior.

Deprecated:
- Legacy flux-threshold knobs (f_lim, F50, w, F50-table, w-table, ra-dec-factor) are removed from the CLI and codepaths.

## Label models

- Physical labels (LAE, OII):
  - `rate_density(row, ctx)`: integrates LF × dV/dz × selection × Jacobians over flux at the candidate’s wavelength.
  - `extra_log_likelihood(row, ctx)`: measurement-only evidence marginalized over latent flux with a neutral prior.
    - Note: `log_evidence(row, ctx)` is deprecated and remains as a shim that delegates to `extra_log_likelihood` while emitting a DeprecationWarning.
- Fake label:
  - `rate_density(row, ctx)`: base fake rate per sr per Å modulated by empirical λ-PDF (if available) and multiplied by effective_search_measure; optionally a simple mixture across components.
  - `extra_log_likelihood(row, ctx)`: measurement-only evidence marginalized over latent flux with a neutral prior.
    - Note: `log_evidence(row, ctx)` is deprecated and remains as a shim that delegates to `extra_log_likelihood` while emitting a DeprecationWarning.

## Caches

- `flux_grid`: a shared quadrature grid (values and log-weights) used to marginalize over latent flux in evidence calculations.
- `fake_lambda_pdf`: optional empirical wavelength PDF cache used to modulate the fake rate prior shape. Built from virtual detections.

## Setting config via CLI

The `simulate` command sets most of these keys directly from its options; see User Guide → CLI for details. When using `classify` directly, defaults are used unless you provide a custom builder.
