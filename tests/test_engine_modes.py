import math
import numpy as np
import pandas as pd

from jlc.engine.engine import JaynesianEngine
from jlc.types import SharedContext


class _StubModel:
    def __init__(self, label, rate_value=1.0, llh_offset=0.0):
        self.label = label
        self._rate_value = float(rate_value)
        self._llh_offset = float(llh_offset)

    # New architecture methods used by the engine
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


def softmax2(a, b):
    ea, eb = math.exp(a), math.exp(b)
    s = ea + eb
    return ea / s, eb / s


def test_engine_mode_rate_only_simple_weights():
    # Two labels with rates 2 and 1, zero likelihoods
    A = _StubModel("A", rate_value=2.0, llh_offset=0.0)
    B = _StubModel("B", rate_value=1.0, llh_offset=0.0)
    reg = _StubRegistry([A, B])
    ctx = _Ctx({"engine_mode": "rate_only"})
    eng = JaynesianEngine(reg, ctx)

    df = pd.DataFrame({"x": [0, 1, 2]})
    out = eng.compute_posteriors(df)

    # Expect constant posteriors across rows: pA=2/3, pB=1/3
    pA = out["p_A"].to_numpy()
    pB = out["p_B"].to_numpy()
    assert np.allclose(pA, 2.0 / 3.0)
    assert np.allclose(pB, 1.0 / 3.0)


def test_engine_mode_likelihood_only_offsets():
    # Equal rates but likelihoods differ by ln 9 => odds 9:1 for B
    A = _StubModel("A", rate_value=1.0, llh_offset=0.0)
    B = _StubModel("B", rate_value=1.0, llh_offset=math.log(9.0))
    reg = _StubRegistry([A, B])
    ctx = _Ctx({"engine_mode": "likelihood_only"})
    eng = JaynesianEngine(reg, ctx)

    df = pd.DataFrame({"x": [0]})
    out = eng.compute_posteriors(df)

    pA, pB = out.loc[0, "p_A"], out.loc[0, "p_B"]
    # Expect pB ~ 0.9, pA ~ 0.1
    assert np.isclose(pB, 0.9, rtol=1e-12, atol=1e-12)
    assert np.isclose(pA, 0.1, rtol=1e-12, atol=1e-12)


def test_engine_mode_combined_rate_times_likelihood():
    # Rates: A=2, B=1. Likelihood offsets: A=0, B=ln 3 => combined weights: A=2, B=3
    A = _StubModel("A", rate_value=2.0, llh_offset=0.0)
    B = _StubModel("B", rate_value=1.0, llh_offset=math.log(3.0))
    reg = _StubRegistry([A, B])
    ctx = _Ctx({"engine_mode": "rate_times_likelihood"})
    eng = JaynesianEngine(reg, ctx)

    df = pd.DataFrame({"x": [0]})
    out = eng.compute_posteriors(df)

    pA, pB = out.loc[0, "p_A"], out.loc[0, "p_B"]
    # Combined proportional weights 2 and 3 => pA=0.4, pB=0.6
    assert np.isclose(pA, 0.4, rtol=1e-12, atol=1e-12)
    assert np.isclose(pB, 0.6, rtol=1e-12, atol=1e-12)
