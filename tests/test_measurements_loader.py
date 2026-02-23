import types

from jlc.utils import load_measurements_from_config
from jlc.priors import PriorRecord


def test_load_measurements_from_config_applies_noise_prior():
    cfg = {
        "measurements": {
            "flux": {
                "module": "jlc.measurements.flux:FluxMeasurement",
                "catalog_columns": {"value": "flux_hat", "error": "flux_err"},
                "latent_key": "F_true",
            }
        }
    }
    rec = PriorRecord(
        name="test_label_prior",
        scope="label",
        label="oii",
        hyperparams={
            "measurements": {
                "flux": {
                    "noise": {
                        "type": "gaussian",
                        "params": {"extra_scatter": 1.23},
                    }
                }
            }
        },
    )
    mods = load_measurements_from_config(cfg, prior_record=rec)
    assert len(mods) == 1
    m = mods[0]
    # Basic type/metadata checks
    from jlc.measurements.flux import FluxMeasurement

    assert isinstance(m, FluxMeasurement)
    assert m.latent_key == "F_true"
    assert tuple(m.catalog_columns) == ("flux_hat", "flux_err")
    # Prior application
    assert isinstance(m.noise_hyperparams, dict)
    assert m.noise_hyperparams.get("extra_scatter") == 1.23
