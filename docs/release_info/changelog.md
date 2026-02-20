# Changelog

All notable changes to this project will be documented in this file.

## Unreleased
- Complete refactor implementation: engine-aligned simulator is sole backend; legacy PPP path removed with erroring shim in CLI.
- Primary rename completed: `log_evidence` → `extra_log_likelihood` across engine/CLI/tests and docs; label shims retained with DeprecationWarning (stacklevel=2) for one minor version.
- Parity harness and batch runner: reproducible PPP parity checks with optional JSON export; reports added under parity_reports/.
- Logging/warnings unified via jlc.utils.logging.log; docstrings normalized across SelectionModel/LabelModel/Engine and label subclasses.
- Add Panacea-like documentation structure with MkDocs.
- Document Phase 1 and Phase 2 features, CLI options, outputs, and configuration.

## Phase 2 (current session)
- Virtual volume mode and empirical fake λ‑PDF calibration utilities.
- Effective search measure knobs and CLI plumbing.
- Fake mixture scaffolding with per-component rate diagnostics.
- Separation of rate priors from data evidences across all labels.

## Phase 1
- Observed‑space rate priors integrated with measurement evidences.
- PPP simulator and expected counts/volume diagnostics.
- Flux plot x-axis logarithmic with lower bound 1e-18.
- Astropy-based cosmology with optional lookup interpolation.
