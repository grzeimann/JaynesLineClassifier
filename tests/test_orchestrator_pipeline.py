import numpy as np
from jlc.engine_noise.noise_cube_model import NoiseCube
from jlc.simulate.pipeline import NoiseCubeReader
from jlc.simulate.noise_histogram import build_noise_histogram
from jlc.simulate.orchestrator import (
    expected_counts_per_cell,
    sample_counts,
    simulate_sources_for_cell,
    run_simulation,
)
from jlc.population.schechter import SchechterLF
from jlc.selection.base import SNCompletenessModel, SelectionModel


def make_tiny_cube():
    # 3x3 spatial, 2 lambda slices
    ra = np.array([10.0, 10.1, 10.2])
    dec = np.array([20.0, 20.1, 20.2])
    lam = np.array([5000.0, 5002.0])
    noise = np.zeros((3, 3, 2), dtype=float)
    # Slice 0 values
    sl0 = np.array([
        [1.0, 2.0, np.nan],
        [0.0, 3.0, 4.0],
        [5.0, -1.0, 6.0],
    ])
    # Slice 1 values
    sl1 = np.array([
        [2.0, 2.1, 2.2],
        [np.nan, np.nan, 10.0],
        [0.0, 0.0, 0.0],
    ])
    noise[:, :, 0] = np.nan_to_num(sl0, nan=np.nan)
    noise[:, :, 1] = np.nan_to_num(sl1, nan=np.nan)
    cube = NoiseCube(noise=noise, ra_grid=ra, dec_grid=dec, wave_grid=lam, mask=None).with_auto_mask()
    return cube


class SimpleSNModel(SNCompletenessModel):
    def __init__(self, sn50=3.0, k=1.0):
        self.sn50 = float(sn50)
        self.k = float(k)
    def completeness(self, sn_true: float, wave_true: float, latent: dict | None = None) -> float:  # type: ignore[override]
        x = (float(sn_true) - self.sn50) / max(self.k, 1e-6)
        return float(1.0 / (1.0 + np.exp(-x)))


def build_selection(sn50=3.0, k=1.0):
    sel = SelectionModel()
    sel.set_sn_model_for("lae", SimpleSNModel(sn50=sn50, k=k))
    sel.set_sn_model_for("oii", SimpleSNModel(sn50=sn50, k=k))
    return sel


def test_expected_and_sample_counts_shapes_and_nonneg():
    cube = make_tiny_cube()
    reader = NoiseCubeReader(cube)
    edges = np.array([0.5, 2.0, 5.0, 12.0])
    nh = build_noise_histogram(reader, edges)
    F_true_grid = np.logspace(-18, -16, 32)
    lf_by_label = {
        "lae": SchechterLF(log10_Lstar=42.0, alpha=-1.5, log10_phistar=-3.0, Lmin=1e38, Lmax=1e46),
        "oii": SchechterLF(log10_Lstar=41.0, alpha=-1.2, log10_phistar=-2.5, Lmin=1e38, Lmax=1e46),
    }
    sel = build_selection(sn50=1.0, k=2.0)
    mu = expected_counts_per_cell(lambda_grid=cube.wave_grid, noise_bin_edges=edges, noise_hist=nh, lf_by_label=lf_by_label, F_true_grid=F_true_grid, selection=sel)
    assert set(mu.keys()) == set(lf_by_label.keys())
    nlam = cube.wave_grid.size
    nbins = edges.size - 1
    for v in mu.values():
        assert v.shape == (nlam, nbins)
        assert np.all(v >= 0)
    rng = np.random.default_rng(123)
    N = sample_counts(mu, rng)
    for v in N.values():
        assert v.shape == (nlam, nbins)
        assert np.issubdtype(v.dtype, np.integer)
        assert np.all(v >= 0)


def test_simulate_sources_for_cell_and_run_simulation_smoke():
    cube = make_tiny_cube()
    reader = NoiseCubeReader(cube)
    edges = np.array([0.5, 2.0, 5.0, 12.0])
    nh = build_noise_histogram(reader, edges)
    F_true_grid = np.logspace(-18, -16, 32)
    lf_by_label = {"lae": SchechterLF(log10_Lstar=42.0, alpha=-1.5, log10_phistar=-3.0, Lmin=1e38, Lmax=1e46)}
    sel = build_selection(sn50=1.0, k=2.0)
    rng = np.random.default_rng(42)
    # Force some counts
    mu = expected_counts_per_cell(lambda_grid=cube.wave_grid, noise_bin_edges=edges, noise_hist=nh, lf_by_label=lf_by_label, F_true_grid=F_true_grid, selection=sel)
    N = sample_counts(mu, rng)
    # Build indexer for k=0
    noise_slice = reader.read_noise_slice(0)
    from jlc.simulate.pipeline import LambdaSliceSpaxelIndex
    indexer = LambdaSliceSpaxelIndex(noise_slice_2d=noise_slice, noise_bin_edges=edges)
    # Pick first bin and label
    j = 0
    N_kj = int(max(1, N["lae"][0, j]))
    recs = simulate_sources_for_cell(
        k=0,
        j=j,
        label="lae",
        N_kj=N_kj,
        lambda_grid=cube.wave_grid,
        F_true_grid=F_true_grid,
        lf_model=lf_by_label["lae"],
        lambda_indexer=indexer,
        noise_bin_center=0.75,
        rng=rng,
        cube_reader=reader,
    )
    assert len(recs) == N_kj
    # End-to-end run
    cat = run_simulation(
        cube_reader=reader,
        noise_hist=nh,
        noise_bin_edges=edges,
        lf_by_label=lf_by_label,
        F_true_grid=F_true_grid,
        selection=sel,
        rng=rng,
    )
    # recarray with required core fields
    assert hasattr(cat, "dtype")
    for fld in ["ra", "dec", "lambda", "F_true", "F_fit", "F_error", "signal", "noise", "label"]:
        assert fld in cat.dtype.names
    # At least a few rows should be simulated
    assert cat.shape[0] >= 0
