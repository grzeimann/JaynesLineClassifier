import numpy as np
from dataclasses import dataclass

from jlc.selection.base import SNCompletenessModel, SelectionModel
from jlc.simulate.completeness_providers import (
    CatalogCompleteness,
    NoiseBinConditionedCompleteness,
    NoiseHistogramCompleteness,
)


class SimpleSNModel(SNCompletenessModel):
    """Monotonic logistic completeness in S/N for testing."""

    def __init__(self, sn50: float = 5.0, k: float = 1.5):
        self.sn50 = float(sn50)
        self.k = float(k)

    def completeness(self, sn_true: float, wave_true: float, latent: dict | None = None) -> float:  # type: ignore[override]
        x = (float(sn_true) - self.sn50) / max(self.k, 1e-6)
        # logistic
        c = 1.0 / (1.0 + np.exp(-x))
        return float(np.clip(c, 0.0, 1.0))


@dataclass
class DummyRow:
    noise: float
    lam: float = 5000.0


def build_selection_with_sn(label: str = "lae", sn50: float = 5.0, k: float = 1.5) -> SelectionModel:
    sel = SelectionModel()
    sel.set_sn_model_for(label, SimpleSNModel(sn50=sn50, k=k))
    return sel


def test_catalog_completeness_monotonic_bounds():
    sel = build_selection_with_sn("lae", sn50=3.0, k=1.0)
    row = DummyRow(noise=2.0)
    prov = CatalogCompleteness(selection=sel, row=row)
    F = np.linspace(0.0, 20.0, 101)
    C = prov.completeness(F, 5000.0, "lae")
    # Bounds
    assert np.all(C >= 0) and np.all(C <= 1)
    # Monotonic non-decreasing in F
    dC = np.diff(C)
    assert np.all(dC >= -1e-12)
    # Low flux should be small; high flux near 1
    assert C[0] <= 0.1
    assert C[-1] >= 0.99


def test_noise_bin_conditioned_completeness_behavior():
    sel = build_selection_with_sn("lae", sn50=5.0, k=2.0)
    prov = NoiseBinConditionedCompleteness(selection=sel, noise_value=1.0)
    F = np.array([0.0, 1.0, 5.0, 10.0])
    C = prov.completeness(F, 6000.0, "lae")
    assert C.shape == F.shape
    assert np.all(C >= 0) and np.all(C <= 1)
    # Higher F gives higher completeness
    assert C[0] <= C[1] <= C[2] <= C[3]


def test_noise_histogram_completeness_marginalizes():
    # Build a tiny histogram proxy with two bins and simple weights
    class DummyHist:
        def __init__(self, centers, weights):
            self.centers = np.asarray(centers, dtype=float)
            self.weights = np.asarray(weights, dtype=float)
        def hist_at_lambda(self, lam):
            # Return fixed centers/weights regardless of lam
            return self.centers, self.weights / (self.weights.sum() if self.weights.sum() > 0 else 1.0), 0

    sel = build_selection_with_sn("lae", sn50=4.0, k=1.0)
    centers = [1.0, 4.0]  # different noise values
    weights = [0.25, 0.75]
    hist = DummyHist(centers, weights)
    prov = NoiseHistogramCompleteness(selection=sel, hist=hist)
    F = np.linspace(0.0, 20.0, 51)
    C = prov.completeness(F, 7000.0, "lae")
    assert C.shape == F.shape
    assert np.all(C >= 0) and np.all(C <= 1)
    # Should be between completeness at min and max noise
    prov_low = NoiseBinConditionedCompleteness(selection=sel, noise_value=centers[0])
    prov_high = NoiseBinConditionedCompleteness(selection=sel, noise_value=centers[1])
    C_low = prov_low.completeness(F, 7000.0, "lae")
    C_high = prov_high.completeness(F, 7000.0, "lae")
    assert np.all(C >= np.minimum(C_low, C_high) - 1e-12)
    assert np.all(C <= np.maximum(C_low, C_high) + 1e-12)
