# Quickstart

This quickstart shows how to classify an existing catalog and how to generate a simulated catalog and classify it.

## Classify an existing catalog

Input CSV must contain at least: `wave_obs`, `flux_hat`, `flux_err` (RA/DEC optional).

Example:

```bash
jlc classify path/to/input.csv --out path/to/output.csv
```

The output includes for each label (default: lae, oii, fake):
- `logZ_<label>`: log evidence from measurement likelihoods (flux-marginalized)
- `p_<label>`: posterior probabilities
- Rate diagnostics: `rate_<label>`, `rate_phys_total`, `rate_fake_total`, `prior_odds_phys_over_fake`

## Simulate and classify (PPP model-driven)

Generate a mock catalog in a sky box and wavelength band using the model-driven Poisson Point Process (PPP):

```bash
jlc simulate --from-model \
  --ra-low 150 --ra-high 150.5 --dec-low 0 --dec-high 0.5 \
  --wave-min 3500 --wave-max 5500 \
  --f-lim 2e-17 --flux-err 1e-17 \
  --fake-rate 2e3 \
  --out-catalog sim_catalog.csv --out-classified sim_classified.csv \
  --plot-prefix sim
```

Notes:
- Expected counts and label-specific comoving volumes are printed to stdout and stored in the context for prior weighting.
- The classifier uses per-row rate priors and may optionally apply global prior weights from PPP expectations during normalization.

## Virtual volume fake-only run

Run a fake-only simulation to calibrate empirical fake λ-intensity:

```bash
jlc simulate --from-model --volume-mode virtual \
  --ra-low 150 --ra-high 150.5 --dec-low 0 --dec-high 0.5 \
  --wave-min 3500 --wave-max 5500 \
  --flux-err 1e-17 \
  --fake-rate 2e3 \
  --out-catalog virtual.csv --out-classified virtual_classified.csv
```

You can then build a λ-PDF from `virtual.csv` and reuse it (see User Guide → Configuration and CLI Usage).
