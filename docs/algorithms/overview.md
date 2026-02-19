# Algorithms and Models

This project implements a Bayesian (Jaynesian) approach to classifying emission-line candidates. The key pieces are:

- Measurement likelihoods (data model): modules that compute log-likelihood of observed quantities given latent variables (e.g., true flux). The default is a Gaussian flux likelihood.
- Population rate priors in observed space: per-row, per-label rate densities r(λ) (per sr per Å) that encode LF × cosmology × selection for physical labels and a contextual mixture rate for the Fake label.
- Posterior computation: log posterior ∝ logZ (measurement-only evidence) + log r (rate prior) + optional global prior weights.

## Physical labels (LAE, OII)

- Redshift from wavelength: z = λ_obs / λ_rest − 1.
- Cosmology: Astropy Planck18 provides d_L(z) in Mpc and dV/dz per sr.
- Flux→Luminosity: L = 4π (d_L[Mpc]·MPC_TO_CM)^2 F (in CGS), with dL/dF Jacobian.
- Selection: completeness S(F, λ_obs) from SelectionModel.
- Observed-space rate: r(λ) = ∫ dF [ dV/dz × φ(L(F,z)) × dL/dF × S(F,λ) × |dz/dλ| ].
- Evidence: marginalize measurement likelihood over F on a shared FluxGrid with neutral prior over F; no population terms included (to avoid double counting).

## Fake label (contextual mixture)

- Base intensity ρ (per sr per Å) configured via CLI; multiplied by an empirical wavelength shape s(λ) learned from virtual detections when available.
- Optional simple mixture over components (e.g., sky_residual, noise) with uniform weights by default; per-component rates are exposed as diagnostics and sum to the total Fake rate.
- Evidence: same measurement-only marginalization over F with a neutral prior.

## Engine and outputs

- The engine computes per-label log evidences `logZ_<label>`, rate priors `rate_<label>`, and combines them (plus optional PPP expected-count weights) to form normalized posteriors `p_<label>`.
- Diagnostics include `rate_phys_total`, `rate_fake_total`, and `prior_odds_phys_over_fake`, and when the Fake mixture is enabled, `rate_fake_<component>` columns.

## Simulation (PPP)

- A Poisson point process generates catalogs where physical source counts follow LF × volume × selection, and fake counts follow a homogeneous rate per sr per Å.
- Expected counts per label and label-specific volumes are computed and printed for diagnostics and stored in the context.
