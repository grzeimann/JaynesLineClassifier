# Examples

This page collects short, copy‑pasteable examples for common workflows.

## 1. Classify a catalog

```bash
jlc classify data/candidates.csv --out results/classified.csv
```

Inspect the top candidates by `p_lae`:

```python
import pandas as pd
out = pd.read_csv('results/classified.csv')
print(out.sort_values('p_lae', ascending=False).head(10))
```

## 2. Simulate with simple fractions

```bash
jlc simulate --n 2000 \
  --ra-low 150 --ra-high 160 --dec-low -2 --dec-high 2 \
  --wave-min 4800 --wave-max 9800 \
  --f-lim 1e-17 --flux-err 5e-18 \
  --lae-frac 0.3 --oii-frac 0.3 --fake-frac 0.4 \
  --out-catalog sim_catalog.csv --out-classified sim_classified.csv \
  --plot-prefix sim
```

## 3. Simulate with the model-driven PPP

```bash
jlc simulate --from-model \
  --ra-low 150 --ra-high 150.5 --dec-low 0 --dec-high 0.5 \
  --wave-min 3500 --wave-max 5500 \
  --f-lim 2e-17 --flux-err 1e-17 \
  --fake-rate 2e3 --nz 256 \
  --out-catalog sim_catalog.csv --out-classified sim_classified.csv \
  --plot-prefix sim
```

## 4. Virtual volume and empirical fake λ‑PDF

1) Generate virtual detections (fake‑only):

```bash
jlc simulate --from-model --volume-mode virtual \
  --ra-low 150 --ra-high 150.5 --dec-low 0 --dec-high 0.5 \
  --wave-min 3500 --wave-max 5500 \
  --flux-err 1e-17 --fake-rate 2e3 \
  --out-catalog virtual.csv --out-classified virtual_classified.csv
```

2) Build and save a λ‑PDF cache, then reuse it:

```bash
jlc simulate --from-model \
  --fake-lambda-calib virtual.csv --fake-lambda-nbins 200 \
  --fake-lambda-cache-out fake_lambda_shape.npz \
  --ra-low 150 --ra-high 151 --dec-low 0 --dec-high 1 \
  --wave-min 3500 --wave-max 5500 \
  --f-lim 2e-17 --flux-err 1e-17 --fake-rate 2e3 \
  --out-catalog sim2.csv --out-classified sim2_classified.csv
```

Or load a precomputed cache directly with `--fake-lambda-cache-in fake_lambda_shape.npz`.

---

## 5. Experimental simulation with a FITS noise cube

Developer feature for end‑to‑end simulations driven by a 3D FITS noise cube. Produces a simulated catalog and optional plots; can also classify the simulated catalog.

Minimal smoke test:

```bash
jlc simulate \
  --sim-pipeline-experimental \
  --noise-cube "/path/to/VDFI_COSMOS_errorcube.fits" \
  --out-catalog cosmos_sim_catalog.csv \
  --seed 12345
```

Recommended (load priors so LFs are available, pin a flux grid, write plots):

```bash
jlc simulate \
  --sim-pipeline-experimental \
  --noise-cube "/path/to/VDFI_COSMOS_errorcube.fits" \
  --load-prior configs/priors \
  --fluxgrid-min 1e-18 --fluxgrid-max 1e-14 --fluxgrid-n 128 \
  --out-catalog cosmos_sim_catalog.csv \
  --plot-prefix cosmos_sim \
  --seed 20260226
```

Optional: classify the experimental catalog via the Engine and write posterior plots:

```bash
jlc simulate \
  --sim-pipeline-experimental \
  --noise-cube "/path/to/VDFI_COSMOS_errorcube.fits" \
  --load-prior configs/priors \
  --fluxgrid-min 1e-18 --fluxgrid-max 1e-14 --fluxgrid-n 128 \
  --out-catalog cosmos_sim_catalog.csv \
  --out-classified cosmos_sim_classified.csv \
  --plot-prefix cosmos_sim \
  --classify-after-experimental \
  --seed 20260226
```

Notes
- Provide `--timing-detail` to enable detailed progress and memory diagnostics.
- When `--plot-prefix` is supplied, the experimental path writes: `<prefix>_wave.png`, `<prefix>_flux.png`, `<prefix>_<label>.png`, `<prefix>_compare.png`, `<prefix>_circle.png`, and `<prefix>_selection.png`.
- See SIMULATION_PIPELINE.md for architecture and design details.
