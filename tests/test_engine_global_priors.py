import numpy as np
import pandas as pd

from jlc.engine.engine import JaynesianEngine


class _StubModel:
    def __init__(self, label, rate_value=1.0, llh_offset=0.0):
        self.label = label
        self._rate_value = float(rate_value)
        self._llh_offset = float(llh_offset)

    def rate_density(self, row, ctx):
        return float(self._rate_value)

    def extra_log_likelihood(self, row, ctx):
        return float(self._llh_offset)


class _StubRegistry:
    def __init__(self, models):
        self._models = {m.label: m for m in models}
        self.labels = list(self._models.keys())

    def model(self, label):
        return self._models[label]


class _Ctx:
    def __init__(self, config=None):
        self.config = dict(config or {})
        self.caches = {}
        self.selection = None
        self.cosmo = None


def test_engine_applies_global_log_prior_weights_when_enabled():
    # Two labels with identical rates and likelihoods, but global priors favor B by ln 9
    A = _StubModel("A", rate_value=1.0, llh_offset=0.0)
    B = _StubModel("B", rate_value=1.0, llh_offset=0.0)
    reg = _StubRegistry([A, B])
    ctx = _Ctx({"engine_mode": "rate_times_likelihood", "use_global_priors": True})
    eng = JaynesianEngine(reg, ctx)

    df = pd.DataFrame({"x": [0]})
    # Provide log_prior_weights equivalent to odds 9:1 for B over A
    log_prior_weights = {"A": 0.0, "B": np.log(9.0)}
    out = eng.compute_posteriors(df, log_prior_weights=log_prior_weights)

    pA, pB = out.loc[0, "p_A"], out.loc[0, "p_B"]
    # Expect pB ~ 0.9, pA ~ 0.1 purely from prior weights
    assert np.isclose(pB, 0.9, rtol=1e-12, atol=1e-12)
    assert np.isclose(pA, 0.1, rtol=1e-12, atol=1e-12)


def test_engine_ignores_global_priors_when_disabled():
    # Same setup but disable use_global_priors
    A = _StubModel("A", rate_value=1.0, llh_offset=0.0)
    B = _StubModel("B", rate_value=1.0, llh_offset=0.0)
    reg = _StubRegistry([A, B])
    ctx = _Ctx({"engine_mode": "rate_times_likelihood", "use_global_priors": False})
    eng = JaynesianEngine(reg, ctx)

    df = pd.DataFrame({"x": [0]})
    log_prior_weights = {"A": 0.0, "B": np.log(9.0)}
    out = eng.compute_posteriors(df, log_prior_weights=log_prior_weights)

    pA, pB = out.loc[0, "p_A"], out.loc[0, "p_B"]
    # With equal rates and likelihoods and global priors disabled, expect 0.5/0.5
    assert np.isclose(pB, 0.5, rtol=1e-12, atol=1e-12)
    assert np.isclose(pA, 0.5, rtol=1e-12, atol=1e-12)
