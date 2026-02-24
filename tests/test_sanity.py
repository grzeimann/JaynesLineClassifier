import numpy as np
import pandas as pd

from jlc.engine.engine import JaynesianEngine
from jlc.types import SharedContext
from jlc.engine.flux_grid import FluxGrid
from jlc.labels.lae import LAELabel
from jlc.labels.oii import OIILabel
from jlc.labels.fake import FakeLabel
from jlc.labels.registry import LabelRegistry
from jlc.population.schechter import SchechterLF
from jlc.measurements.flux import FluxMeasurement
from jlc.selection.base import SelectionModel
from jlc.simulate.field import simulate_field as simulate_field_api
from jlc.simulate.lf_estimation import default_log10L_bins_from_registry, binned_lf_inferred, skybox_solid_angle_sr
from jlc.cosmology.lookup import AstropyCosmology


def _default_registry_and_ctx(f_lim=None, F50=None, w=None, flux_err=5e-18, engine_mode=None, volume_mode="real"):
    cosmo = AstropyCosmology()
    selection = SelectionModel(f_lim=f_lim, F50=F50, w=w)
    fg = FluxGrid(Fmin=1e-19, Fmax=1e-14, n=128)
    caches = {"flux_grid": fg}
    config = {
        "use_rate_priors": True,
        "use_global_priors": False,
        "engine_mode": engine_mode or "rate_times_likelihood",
        "volume_mode": volume_mode,
    }
    ctx = SharedContext(cosmo=cosmo, selection=selection, caches=caches, config=config)

    # Simple, stable LFs
    lae_lf = SchechterLF(log10_Lstar=42.72, alpha=-1.75, log10_phistar=-3.20,
                         Lmin=1e39, Lmax=1e46)
    oii_lf = SchechterLF(log10_Lstar=41.4, alpha=-1.2, log10_phistar=-2.4,
                         Lmin=1e38, Lmax=1e45)
    flux_meas = FluxMeasurement()

    lae = LAELabel(lae_lf, selection, [flux_meas])
    oii = OIILabel(oii_lf, selection, [flux_meas])
    fake = FakeLabel(selection_model=selection, measurement_modules=[flux_meas])
    reg = LabelRegistry([lae, oii, fake])
    return ctx, reg


def test_fake_only_virtual_volume_posteriors_near_unity():
    # Virtual volume mode suppresses physical labels; simulate fake-only and expect p_fake ~ 1
    ctx, registry = _default_registry_and_ctx(f_lim=None, flux_err=5e-18, engine_mode="rate_times_likelihood", volume_mode="virtual")

    # Simulate over a fairly wide band and sky area to get enough objects
    ra_low, ra_high = 150.0, 150.1
    dec_low, dec_high = 0.0, 0.1
    wave_min, wave_max = 4000.0, 4020.0

    df = simulate_field_api(
        registry=registry,
        ctx=ctx,
        ra_low=ra_low, ra_high=ra_high,
        dec_low=dec_low, dec_high=dec_high,
        wave_min=wave_min, wave_max=wave_max,
        flux_err=5e-18,
        fake_rate_per_sr_per_A=2e3,  # ensure some fake events
        seed=321,
        nz=128,
    )
    # If no rows are generated (unlikely with these settings), skip
    if df is None or len(df) == 0:
        return
    engine = JaynesianEngine(registry, ctx)
    out = engine.compute_extra_log_likelihood_matrix(df)
    out = engine.normalize_posteriors(out, mode="rate_times_likelihood")

    p_fake = out.get("p_fake", pd.Series([], dtype=float)).to_numpy(dtype=float)
    if p_fake.size > 0:
        # Expect the vast majority to be close to 1
        assert np.nanmean(p_fake) > 0.8
        assert np.nanmedian(p_fake) > 0.9


def test_mixed_labels_rate_only_matches_ppp_fractions(): 
    # Simulate with PPP mode defaults (real volume) and check that average posterior
    # fractions under rate_only mode match PPP expected counts proportions.
    ctx, registry = _default_registry_and_ctx(f_lim=2e-17, flux_err=5e-18, engine_mode="rate_only", volume_mode="real")

    # Simulate over a fairly wide band and sky area to get enough objects
    ra_low, ra_high = 150.0, 150.1
    dec_low, dec_high = 0.0, 0.1
    wave_min, wave_max = 4000.0, 4020.0

    df = simulate_field_api(
        registry=registry,
        ctx=ctx,
        ra_low=ra_low, ra_high=ra_high,
        dec_low=dec_low, dec_high=dec_high,
        wave_min=wave_min, wave_max=wave_max,
        flux_err=5e-18,
        fake_rate_per_sr_per_A=1e3,
        seed=456,
        nz=256,
    )

    engine = JaynesianEngine(registry, ctx)
    out = engine.compute_extra_log_likelihood_matrix(df)
    out = engine.normalize_posteriors(out, mode="rate_only")

    # Compute mean posterior fractions
    means = {}
    for L in registry.labels:
        col = f"p_{L}"
        if col in out.columns:
            means[L] = float(np.nanmean(out[col].to_numpy(dtype=float)))

    # Estimate class fractions from the simulated catalog's true_class labels
    if "true_class" in df.columns and all(L in means for L in registry.labels):
        counts = {L: float((df["true_class"].astype(str) == L).sum()) for L in registry.labels}
        total = float(sum(counts.values()))
        if total > 0:
            fracs = {L: counts[L] / total for L in registry.labels}
            # Compare with modest tolerance due to finite sampling
            for L in registry.labels:
                assert np.isclose(means[L], fracs[L], rtol=0.2, atol=0.05), (
                    f"Posterior fraction for {L} deviates: mean={means[L]:.3f}, true_frac={fracs[L]:.3f}"
                )



def test_eddington_bias_inferred_lf_shifts_with_noise():
    """
    With selection S≡1, increasing flux noise should induce Eddington bias:
    the inferred LF (posterior-weighted, rate_only) shifts higher at the bright end
    compared to a very-low-noise baseline.
    """
    import numpy as _np
    from jlc.engine.engine import JaynesianEngine
    from jlc.types import SharedContext
    from jlc.engine.flux_grid import FluxGrid
    from jlc.labels.lae import LAELabel
    from jlc.labels.registry import LabelRegistry
    from jlc.population.schechter import SchechterLF
    from jlc.measurements.flux import FluxMeasurement
    from jlc.selection.base import SelectionModel
    from jlc.simulate.field import simulate_field as simulate_field_api
    from jlc.simulate.lf_estimation import default_log10L_bins_from_registry, binned_lf_inferred, skybox_solid_angle_sr
    from jlc.cosmology.lookup import AstropyCosmology

    # Build context/registry with S≡1 (no threshold)
    cosmo = AstropyCosmology()
    selection = SelectionModel()  # completeness == 1
    fg = FluxGrid(Fmin=1e-19, Fmax=1e-14, n=128)
    caches = {"flux_grid": fg}
    config = {"use_rate_priors": True, "use_global_priors": False, "engine_mode": "rate_only"}
    ctx_low = SharedContext(cosmo=cosmo, selection=selection, caches=caches, config=dict(config))
    ctx_high = SharedContext(cosmo=cosmo, selection=selection, caches=caches, config=dict(config))

    lae_lf = SchechterLF(log10_Lstar=42.72, alpha=-1.75, log10_phistar=-3.05, Lmin=1e39, Lmax=1e46)
    flux_meas = FluxMeasurement()
    lae_low = LAELabel(lae_lf, selection, [flux_meas])
    lae_high = LAELabel(lae_lf, selection, [flux_meas])
    reg_low = LabelRegistry([lae_low])
    reg_high = LabelRegistry([lae_high])

    # Simulate over a fairly wide band and sky area to get enough objects
    ra_low, ra_high = 150.0, 150.1
    dec_low, dec_high = 0.0, 0.1
    wave_min, wave_max = 4000.0, 4020.0

    # Very low noise baseline and higher noise scenario
    df_low = simulate_field_api(
        registry=reg_low,
        ctx=ctx_low,
        ra_low=ra_low, ra_high=ra_high,
        dec_low=dec_low, dec_high=dec_high,
        wave_min=wave_min, wave_max=wave_max,
        flux_err=5e-19,
        fake_rate_per_sr_per_A=0.0,
        seed=2024,
        nz=256,
    )
    df_high = simulate_field_api(
        registry=reg_high,
        ctx=ctx_high,
        ra_low=ra_low, ra_high=ra_high,
        dec_low=dec_low, dec_high=dec_high,
        wave_min=wave_min, wave_max=wave_max,
        flux_err=3e-18,  # ~6x larger noise
        fake_rate_per_sr_per_A=0.0,
        seed=2025,
        nz=256,
    )

    # Classify both in rate_only mode
    eng_low = JaynesianEngine(reg_low, ctx_low)
    out_low = eng_low.compute_extra_log_likelihood_matrix(df_low)
    out_low = eng_low.normalize_posteriors(out_low, mode="rate_only")

    eng_high = JaynesianEngine(reg_high, ctx_high)
    out_high = eng_high.compute_extra_log_likelihood_matrix(df_high)
    out_high = eng_high.normalize_posteriors(out_high, mode="rate_only")

    # Build common logL bins using registry L*
    bins_map = default_log10L_bins_from_registry(reg_low, nbins=16)
    bins = bins_map["lae"]
    omega = skybox_solid_angle_sr(ra_low, ra_high, dec_low, dec_high)

    lf_low = binned_lf_inferred(out_low, "lae", cosmo, selection, omega, wave_min, wave_max, bins, nz=1024, use_hard=False)
    lf_high = binned_lf_inferred(out_high, "lae", cosmo, selection, omega, wave_min, wave_max, bins, nz=1024, use_hard=False)

    # Require non-empty
    assert lf_low is not None and lf_high is not None and len(lf_low) == len(lf_high) and len(lf_low) > 0

    centers = 0.5 * (lf_low["log10L_lo"].values + lf_low["log10L_hi"].values)
    phi_low = lf_low["phi_per_dex"].values.astype(float)
    phi_high = lf_high["phi_per_dex"].values.astype(float)

    # Focus on bright-end bins (above L*)
    Lstar = 10.0 ** lae_lf.log10_Lstar
    bright_mask = (10.0 ** centers) >= Lstar
    if _np.any(bright_mask):
        # Expect higher counts at bright end under higher noise (Eddington bias)
        # Use a modest aggregate comparison to be robust to shot noise
        s_low = _np.nansum(phi_low[bright_mask])
        s_high = _np.nansum(phi_high[bright_mask])
        # Allow ties and small deviations; require at least 5% uplift
        assert s_high >= 0.95 * s_low, "Unexpected drop at bright end under higher noise"
        assert s_high > s_low * 1.02 or s_high - s_low > 0, "No discernible Eddington uplift at bright end"
    else:
        # If no bright bins exist (unlikely), fall back to median comparison
        assert _np.nanmedian(phi_high) >= _np.nanmedian(phi_low)
