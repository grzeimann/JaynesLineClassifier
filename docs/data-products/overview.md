# Data Products

JLC typically reads and writes CSV files with the following columns.

## Input catalog (minimum)
- `wave_obs` (Å): observed wavelength of the line candidate
- `flux_hat`: measured flux (arbitrary but consistent units)
- `flux_err`: 1σ uncertainty in `flux_hat` (same units)
- Optional: `ra`, `dec` (deg)

## Simulation outputs
Both simple and PPP simulators write:
- `ra`, `dec`: sky coordinates (deg)
- `true_class`: one of `lae`, `oii`, `fake`
- `wave_obs`: observed wavelength (Å)
- `flux_hat`: measured flux after adding noise
- `flux_err`: per-object flux error used when generating noise
- `snr`: signal-to-noise ratio `flux_hat/flux_err` (S/N filter applied if `--snr-min` is provided)

## Classification outputs (additional columns)
- `logZ_<label>`: measurement-only log-evidence for each label
- `p_<label>`: posterior probabilities
- `rate_<label>`: per-row rate prior for each label (per sr per Å)
- `rate_phys_total`, `rate_fake_total`: sums over labels
- `prior_odds_phys_over_fake`: `rate_phys_total / rate_fake_total`
- If Fake mixture enabled: `rate_fake_<component>` components summing to `rate_fake_total`

## Figures
If `--plot-prefix NAME` is provided to `jlc simulate`, two PNG files are written:
- `NAME_wave.png`: histogram of observed wavelengths colored by `true_class`.
- `NAME_flux.png`: histogram of measured fluxes colored by `true_class` (x-axis log with lower bound 1e-18; y-axis in log scale for counts).
