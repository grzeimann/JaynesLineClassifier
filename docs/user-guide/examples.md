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
