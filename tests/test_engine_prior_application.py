import pandas as pd

from jlc.cli.main import build_default_context_and_registry
from jlc.engine.engine import JaynesianEngine
from jlc.priors import PriorRecord


def test_engine_apply_prior_record_updates_hyperparams_and_measurements():
    # Build minimal context and registry
    ctx, registry = build_default_context_and_registry()
    engine = JaynesianEngine(registry, ctx)

    # Construct a prior record targeting "lae" with population and measurement noise params
    record = PriorRecord(
        name="lae_test_prior",
        scope="label",
        label="lae",
        hyperparams={
            "population": {"log10_Lstar": 42.8, "alpha": -1.7, "log10_phistar": -3.1},
            "measurements": {
                "flux": {
                    "noise": {"type": "gaussian", "params": {"extra_scatter": 1e-18}},
                },
            },
        },
        source="unit_test",
    )

    # Apply prior via engine helper
    engine.apply_prior_record(record)

    # Assert label hyperparameters updated
    lae = registry.model("lae")
    hp = lae.get_hyperparams_dict()
    assert abs(hp["log10_Lstar"] - 42.8) < 1e-12
    assert abs(hp["alpha"] + 1.7) < 1e-12
    assert abs(hp["log10_phistar"] + 3.1) < 1e-12

    # Assert flux measurement noise hyperparams updated
    flux_mod = None
    for m in lae.measurement_modules:
        if getattr(m, "name", None) == "flux":
            flux_mod = m
            break
    assert flux_mod is not None
    assert isinstance(getattr(flux_mod, "noise_hyperparams", None), dict)
    assert abs(flux_mod.noise_hyperparams.get("extra_scatter", 0.0) - 1e-18) < 1e-30
