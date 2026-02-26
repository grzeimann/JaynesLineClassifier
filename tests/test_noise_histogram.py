import numpy as np
from jlc.engine_noise.noise_cube_model import NoiseCube
from jlc.simulate.pipeline import NoiseCubeReader
from jlc.simulate.noise_histogram import build_noise_histogram


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
    # Mask: None; rely on auto invalidation of <=0 and NaN
    cube = NoiseCube(noise=noise, ra_grid=ra, dec_grid=dec, wave_grid=lam, mask=None).with_auto_mask()
    return cube


def test_build_noise_histogram_counts_and_area():
    cube = make_tiny_cube()
    reader = NoiseCubeReader(cube)
    # Choose edges to bin 1..6 into unit bins and also capture 10
    edges = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 11.0])
    nh = build_noise_histogram(reader, edges)

    # Verify lambda grid passthrough
    assert np.allclose(nh.lambda_grid, cube.wave_grid)
    assert nh.counts.shape == (cube.wave_grid.size, edges.size - 1)

    # Manually compute valid counts in slice 0
    # Valid: 1,2,3,4,5,6 (ignore nan, 0.0, -1.0)
    expected0 = np.array([1, 1, 1, 1, 1, 1, 0])
    assert np.array_equal(nh.counts[0], expected0)

    # Slice 1 valid: 2.0,2.1,2.2,10.0 (ignore nans and zeros)
    # Bins: [0.5-1.5)=0, [1.5-2.5)=3, ..., last bin includes 10.0
    expected1 = np.array([0, 3, 0, 0, 0, 0, 1])
    assert np.array_equal(nh.counts[1], expected1)

    # Weights at lambda ~5000 should normalize to 1
    centers0, w0, k0 = nh.hist_at_lambda(5000.0)
    assert k0 == 0
    assert np.isclose(w0.sum(), 1.0)
    # And at lambda ~5002
    _, w1, k1 = nh.hist_at_lambda(5002.0)
    assert k1 == 1
    assert np.isclose(w1.sum(), 1.0)

    # Survey area per slice should be non-negative; slice 0 has more valid spaxels than slice 1
    # Slice 0 valid count = 6; slice 1 valid count = 4; area is uniform per spaxel up to cos(dec) row factor
    assert nh.survey_area_sr.shape == (2,)
    assert np.all(nh.survey_area_sr >= 0)
    assert nh.survey_area_sr[0] > nh.survey_area_sr[1]
