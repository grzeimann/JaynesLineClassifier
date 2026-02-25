import numpy as np
from jlc.engine_noise.noise_cube_model import NoiseCube, NoiseCubeModel


def make_tiny_cube():
    # 2x2x3 cube with simple increasing values, one masked voxel set to 0
    ra = np.array([10.0, 11.0])
    dec = np.array([20.0, 21.0])
    wave = np.array([5000.0, 6000.0, 7000.0])
    noise = np.array([
        [[1.0, 2.0, 3.0],
         [4.0, 5.0, 0.0]],   # last is invalid (0)
        [[7.0, 8.0, 9.0],
         [10.0, 11.0, 12.0]],
    ])
    cube = NoiseCube(noise=noise, ra_grid=ra, dec_grid=dec, wave_grid=wave).with_auto_mask()
    return cube


def test_index_and_value_at_nearest_neighbor():
    cube = make_tiny_cube()
    # A coordinate near first grid points should map to index 0
    ira, idec, ilam = cube.index_of(10.1, 20.2, 5050.0)
    assert (ira, idec, ilam) == (0, 0, 0)
    v = cube.value_at(10.1, 20.2, 5050.0)
    assert np.isfinite(v) and v == 1.0

    # Midpoints should choose the nearest
    ira, idec, ilam = cube.index_of(10.9, 20.9, 6500.0)
    assert (ira, idec, ilam) == (1, 1, 1)
    v = cube.value_at(10.9, 20.9, 6500.0)
    assert np.isfinite(v) and v == 11.0


def test_invalid_masked_voxel_returns_nan():
    cube = make_tiny_cube()
    # Coordinates mapping to the zero entry (0,1,2)
    v, m = cube.value_at(10.0, 21.0, 7000.0, return_mask=True)
    assert m is True
    assert not np.isfinite(v)


def test_noise_cube_model_sigma():
    cube = make_tiny_cube()
    model = NoiseCubeModel(cube)
    # Without RA/Dec returns default
    s = model.sigma(6000.0)
    assert np.isfinite(s) and s == model.default_sigma

    # With position returns the cube value
    s = model.sigma(6000.0, ra=10.0, dec=21.0)
    assert np.isfinite(s) and s == 5.0

    # Invalid voxel should return NaN
    s = model.sigma(7000.0, ra=10.0, dec=21.0)
    assert not np.isfinite(s)
