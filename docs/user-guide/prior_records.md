# Prior Records (Priors & Measurements)

This page explains how to persist and reuse priors for labels and measurements using PriorRecord, and how to use these records from the CLI.

## What is a PriorRecord?

A PriorRecord is a YAML (or JSON) file that stores hyperparameters with provenance metadata. For label scope, it typically contains:
- population: label hyperparameters (e.g., Schechter LF or fake prior parameters)
- measurements: per-measurement priors and noise model hyperparameters keyed by measurement name (e.g., flux, wavelength)
- selection: optional selection-related hyperparams (reserved; current SelectionModel reads directly from CLI/config)

Minimal dataclass (see jlc.priors.record):
- name, scope, label
- hyperparams: dict with optional blocks population, measurements, selection
- source, notes, created_at

## Example YAML (OII)

```yaml
name: "oii_default_v1"
scope: "label"
label: "oii"
source: "literature+initial_em"
notes: "OII LF + basic measurement priors"

hyperparams:
  population:
    log10_Lstar: 41.5
    alpha: -1.3
    log10_phistar: -2.5

  measurements:
    flux:
      noise:
        type: gaussian
        params:
          extra_scatter: 0.0
    wavelength:
      prior:
        type: from_cosmology
        params:
          rest_wave: 3727.0
      noise:
        type: gaussian
        params:
          extra_scatter: 0.0
```

## Built-in example priors

Starter PriorRecord YAMLs are included in the repo under:
- configs/priors/prior_lae.yaml
- configs/priors/prior_oii.yaml
- configs/priors/prior_fake.yaml

These are conservative defaults matching the current code’s assumptions. You can copy and edit them for experiments (e.g., change flux extra_scatter).

## CLI usage

The CLI can load a PriorRecord before running and can snapshot one afterwards.

- classify:
  - --load-prior prior.yaml
  - --save-prior-dir priors_out/
- simulate:
  - --load-prior prior.yaml
  - --save-prior-dir priors_out/

When loaded, population hyperparameters shallow-merge into the matching label’s hyperparams. Measurement blocks update the measurement modules’ noise_hyperparams and prior_hyperparams by name.

Snapshots written to --save-prior-dir capture:
- population: current label hyperparams
- measurements: current noise/prior params for each module (e.g., extra_scatter)

## Programmatic usage

You can also apply a PriorRecord in code via the engine helper:

```python
from jlc.priors import load_prior_record
from jlc.cli.main import build_default_context_and_registry
from jlc.engine.engine import JaynesianEngine

ctx, registry = build_default_context_and_registry()
engine = JaynesianEngine(registry, ctx)
rec = load_prior_record("configs/priors/prior_oii.yaml")
engine.apply_prior_record(rec)
# Inspect updated hyperparams
print(registry.model("oii").get_hyperparams_dict())
```

## Quick testing recipes

- Generate a small mock catalog and save a snapshot of priors actually used:

```bash
jlc simulate \
  --ra-low 150 --ra-high 150.5 --dec-low 0 --dec-high 0.5 \
  --wave-min 4800 --wave-max 9800 \
  --f-lim 1e-17 --flux-err 5e-18 \
  --fake-rate 2e3 \
  --out-catalog sim_catalog.csv \
  --out-classified sim_classified.csv \
  --save-prior-dir priors_out/
```

- Classify an existing catalog while loading an example prior:

```bash
jlc classify input.csv --out output.csv \
  --load-prior configs/priors/prior_oii.yaml
```

- See top slow tests and run the unit tests:

```bash
python -m pip install -e .[test]
pytest --durations=10
```
