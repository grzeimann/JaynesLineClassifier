import numpy as np
from jlc.priors.record import PriorRecord
from jlc.simulate.kernel import (
    GaussianLineProfile,
    SkewGaussianLineProfile,
    OIIDoubletProfile,
    KernelEnv,
    draw_signal_and_flux,
    build_profile_from_prior,
)


def test_build_profile_from_prior_gaussian_and_skew():
    # Gaussian
    rec_g = PriorRecord(
        name="lae_prof_gauss",
        scope="label",
        label="lae",
        hyperparams={
            "measurements": {
                "flux": {
                    "profile": {
                        "type": "gaussian",
                        "params": {"sigma_A": 3.0, "gain": 1.2},
                    }
                }
            }
        },
    )
    p = build_profile_from_prior(rec_g)
    assert isinstance(p, GaussianLineProfile)
    assert np.isclose(p.sigma_A, 3.0)
    assert np.isclose(p.gain, 1.2)

    # Skew Gaussian
    rec_s = PriorRecord(
        name="lae_prof_skew",
        scope="label",
        label="lae",
        hyperparams={
            "measurements": {
                "flux": {
                    "profile": {
                        "type": "skew_gaussian",
                        "params": {"sigma_A": 2.5, "gain": 1.0, "skew": 0.3},
                    }
                }
            }
        },
    )
    p2 = build_profile_from_prior(rec_s)
    assert isinstance(p2, SkewGaussianLineProfile)
    assert np.isclose(p2.skew, 0.3)


def test_profiles_affect_signal_deterministically():
    rng = np.random.default_rng(0)
    env = KernelEnv(lam=6000.0, noise=5e-18)
    F = 1e-17

    g = GaussianLineProfile(sigma_A=2.0, gain=1.0)
    s = SkewGaussianLineProfile(sigma_A=2.0, gain=1.0, skew=0.5)

    sig_g, F_fit_g, F_err_g = draw_signal_and_flux(F, env.lam, env, rng, profile=g)
    sig_s, F_fit_s, F_err_s = draw_signal_and_flux(F, env.lam, env, rng, profile=s)

    assert np.isfinite(sig_g) and np.isfinite(sig_s)
    # With positive skew, signal should be >= gaussian case at the same F
    assert sig_s >= sig_g - 1e-30
    # Flux error reflects the environment noise (no extra scatter)
    assert np.isclose(F_err_g, env.noise)
    assert np.isclose(F_err_s, env.noise)


def test_oii_doublet_overlap_effect():
    rng = np.random.default_rng(1)
    env = KernelEnv(lam=7500.0, noise=1e-17)
    F = 5e-17

    # Same total flux; smaller separation should increase effective signal overlap
    p_wide = OIIDoubletProfile(sigma_A=2.0, gain=1.0, sep_A=6.0, ratio=1.0, filter_sigma_A=2.0)
    p_narrow = OIIDoubletProfile(sigma_A=2.0, gain=1.0, sep_A=1.0, ratio=1.0, filter_sigma_A=2.0)

    sig_w, *_ = draw_signal_and_flux(F, env.lam, env, rng, profile=p_wide)
    sig_n, *_ = draw_signal_and_flux(F, env.lam, env, rng, profile=p_narrow)

    assert sig_n >= sig_w - 1e-30
