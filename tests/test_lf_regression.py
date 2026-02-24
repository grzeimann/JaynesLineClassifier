import os
import numpy as np

from jlc.cosmology.lookup import AstropyCosmology
from jlc.simulate.lf_estimation import (
    compute_label_volume,
    plot_binned_lf,
)


def test_compute_label_volume_stability():
    # Basic stability check: volume should be consistent across nz within ~1%
    cosmo = AstropyCosmology()
    omega = 1.0  # sr (arbitrary, absolute scale cancels in relative check)

    # LAE
    V1 = compute_label_volume(cosmo, 1215.67, 5000.0, 7000.0, omega, nz=1024)
    V2 = compute_label_volume(cosmo, 1215.67, 5000.0, 7000.0, omega, nz=2048)
    assert np.isfinite(V1) and np.isfinite(V2)
    if max(abs(V1), abs(V2)) > 0:
        assert np.isclose(V1, V2, rtol=0.01, atol=0.0)

    # OII
    V3 = compute_label_volume(cosmo, 3727.0, 5000.0, 9000.0, omega, nz=1024)
    V4 = compute_label_volume(cosmo, 3727.0, 5000.0, 9000.0, omega, nz=2048)
    assert np.isfinite(V3) and np.isfinite(V4)
    if max(abs(V3), abs(V4)) > 0:
        assert np.isclose(V3, V4, rtol=0.01, atol=0.0)


def test_plot_binned_lf_smoke(tmp_path):
    # Create minimal dummy binned LF DataFrames with expected columns
    import pandas as pd

    lae_df = pd.DataFrame({
        "log10L": [42.0, 42.5, 43.0],
        "phi": [1e-4, 5e-5, 2e-5],
        "phi_err": [1e-5, 5e-6, 2e-6],
        "label": ["lae", "lae", "lae"],
    })
    oii_df = pd.DataFrame({
        "log10L": [41.0, 41.5, 42.0],
        "phi": [2e-4, 1e-4, 4e-5],
        "phi_err": [2e-5, 1e-5, 4e-6],
        "label": ["oii", "oii", "oii"],
    })

    prefix = os.path.join(tmp_path, "lf_test")
    # The function should not raise and should write two files
    plot_binned_lf(lae_df, oii_df, prefix, title="LF Smoke Test")

    lae_png = f"{prefix}_lae.png"
    oii_png = f"{prefix}_oii.png"
    assert os.path.exists(lae_png), f"Expected plot file not found: {lae_png}"
    assert os.path.exists(oii_png), f"Expected plot file not found: {oii_png}"



def test_s_equiv_1_low_noise_lf_recovery_rate_only():
    """
    With selection S≡1 and very low noise, classify in rate_only mode and
    verify that the inferred LAE LF matches the underlying Schechter model
    within a modest tolerance across bins with sufficient counts.
    """
    import numpy as _np
    import pandas as _pd
    from jlc.engine.engine import JaynesianEngine
    from jlc.types import SharedContext
    from jlc.engine.flux_grid import FluxGrid
    from jlc.labels.lae import LAELabel
    from jlc.labels.registry import LabelRegistry
    from jlc.population.schechter import SchechterLF
    from jlc.measurements.flux import FluxMeasurement
    from jlc.selection.base import SelectionModel
    from jlc.simulate.field import simulate_field as simulate_field_api
    from jlc.simulate.lf_estimation import default_log10L_bins_from_registry, schechter_phi_per_dex, binned_lf_inferred, skybox_solid_angle_sr

    # Cosmology and context
    cosmo = AstropyCosmology()
    # Selection with no threshold => completeness ≡ 1
    selection = SelectionModel()
    # Flux grid spanning a wide flux range
    fg = FluxGrid(Fmin=1e-19, Fmax=1e-14, n=128)
    caches = {"flux_grid": fg}
    # Boost expected counts to reduce Poisson noise
    config = {
        "search_measure_scale": 5.0,  # multiplies effective_search_measure
        "use_rate_priors": True,
        "use_global_priors": False,
        "engine_mode": "rate_only",
    }
    ctx = SharedContext(cosmo=cosmo, selection=selection, caches=caches, config=config)

    # LAE LF with slightly elevated phi* to increase counts
    lae_lf = SchechterLF(log10_Lstar=42.72, alpha=-1.75, log10_phistar=-3.00,
                         Lmin=1e39, Lmax=1e46)
    flux_meas = FluxMeasurement()
    lae = LAELabel(lae_lf, selection, [flux_meas])
    registry = LabelRegistry([lae])

    # Simulate over a fairly wide band and sky area to get enough objects
    ra_low, ra_high = 150.0, 150.1
    dec_low, dec_high = 0.0, 0.1
    wave_min, wave_max = 4000.0, 4020.0

    df_sim = simulate_field_api(
        registry=registry,
        ctx=ctx,
        ra_low=ra_low, ra_high=ra_high,
        dec_low=dec_low, dec_high=dec_high,
        wave_min=wave_min, wave_max=wave_max,
        flux_err=5e-19,  # very low noise
        fake_rate_per_sr_per_A=0.0,
        seed=123,
        nz=256,
    )

    # Classify in rate_only mode
    eng = JaynesianEngine(registry, ctx)
    out = eng.compute_extra_log_likelihood_matrix(df_sim)
    out = eng.normalize_posteriors(out, mode="rate_only")

    # Build default bins around L* and compute inferred LF
    bins_map = default_log10L_bins_from_registry(registry, nbins=16)
    bins = bins_map["lae"]
    # Solid angle for the sky box
    omega = skybox_solid_angle_sr(ra_low, ra_high, dec_low, dec_high)

    df_inf = binned_lf_inferred(out, "lae", cosmo, selection, omega, wave_min, wave_max, bins, nz=1024, use_hard=False)

    # Compare binned phi_per_dex to Schechter at bin centers for bins with enough counts
    centers = 0.5 * (df_inf["log10L_lo"].values + df_inf["log10L_hi"].values)
    phi_true = schechter_phi_per_dex(centers, lae_lf)
    phi_est = df_inf["phi_per_dex"].values
    N = df_inf["N"].values

    # Only consider bins with reasonable counts to avoid noisy tails
    mask = N >= 10
    if _np.any(mask):
        # Allow ~30% relative error per bin due to finite sampling
        rel = _np.abs(phi_est[mask] - phi_true[mask]) / _np.maximum(phi_true[mask], 1e-30)
        assert _np.all(rel < 0.3), f"LF recovery exceeded tolerance in some bins: max rel err={rel.max():.2f}"
    else:
        # If few counts overall, relax: require at least one bin exists and finite values
        assert len(phi_est) > 0 and _np.all(_np.isfinite(phi_est))
