import numpy as np
from jlc.simulate.kernel import GaussianLineProfile, KernelEnv, draw_signal_and_flux


def test_draw_signal_and_flux_noise_scaling_rng_variance():
    rng = np.random.default_rng(123)
    F_true = 1.0e-17
    lam = 6000.0
    sigma = 5.0e-18
    env = KernelEnv(lam=lam, noise=sigma)
    prof = GaussianLineProfile(sigma_A=2.0, gain=1.0)

    # Draw many samples to estimate empirical std of F_fit
    n = 5000
    draws = []
    for _ in range(n):
        _, F_fit, F_err = draw_signal_and_flux(F_true, lam, env, rng, profile=prof)
        # F_error should be close to sigma
        assert np.isfinite(F_err) and F_err >= 0
        draws.append(F_fit)
    draws = np.asarray(draws)
    emp_std = draws.std(ddof=1)
    # Allow 20% tolerance due to finite sampling
    assert np.isclose(emp_std, sigma, rtol=0.2, atol=0.0)


def test_profile_signal_monotonic_in_flux():
    rng = np.random.default_rng(42)
    lam = 5000.0
    sigma = 1.0e-18
    env = KernelEnv(lam=lam, noise=sigma)
    prof = GaussianLineProfile(sigma_A=2.0, gain=1.0)

    F_vals = np.array([1e-19, 5e-19, 1e-18, 5e-18, 1e-17])
    sig_prev = -np.inf
    for F in F_vals:
        sig, F_fit, F_err = draw_signal_and_flux(F, lam, env, rng, profile=prof)
        assert np.isfinite(sig)
        assert sig >= sig_prev - 1e-30  # non-decreasing
        sig_prev = sig
        # F_error should equal input noise (since extra_scatter=0)
        assert np.isclose(F_err, sigma)
