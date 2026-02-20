import pandas as pd
from jlc.utils.logging import log

# For Phase 1, reuse the robust PPP implementation while establishing a
# unified entry point that aligns with the refactor plan. This allows the
# CLI (or callers) to opt into the new simulate_field() API without changing
# underlying behavior, keeping risk minimal.

def simulate_field(
    registry,
    ctx,
    ra_low: float,
    ra_high: float,
    dec_low: float,
    dec_high: float,
    wave_min: float,
    wave_max: float,
    flux_err: float = 1e-17,
    f_lim: float | None = None,
    fake_rate_per_sr_per_A: float = 0.0,
    seed: int | None = None,
    nz: int = 256,
    snr_min: float | None = None,
) -> pd.DataFrame:
    """Unified simulator entry that uses per-label simulate_catalog() methods.

    This is the engine-aligned path and the source of truth going forward.
    It preserves the public signature used by the CLI and callers.
    """
    import numpy as np
    from jlc.simulate.model_ppp import (
        skybox_solid_angle_sr as _omega,
    )

    rng = np.random.default_rng(seed)

    # Record common knobs in context for traceability
    try:
        if isinstance(ctx.config, dict):
            ctx.config["f_lim"] = f_lim if f_lim is not None else ctx.config.get("f_lim", None)
            ctx.config["fake_rate_per_sr_per_A"] = float(fake_rate_per_sr_per_A)
            ctx.config["wave_min"] = float(wave_min)
            ctx.config["wave_max"] = float(wave_max)
            if snr_min is not None:
                ctx.config["snr_min"] = float(snr_min)
            ctx.config["nz_simulator"] = int(nz)
    except Exception:
        pass

    # Solid angle of sky box
    omega = _omega(ra_low, ra_high, dec_low, dec_high)

    # Run per-label simulators and collect diagnostics
    dfs = []
    exp_counts = {"total": 0.0}
    label_volumes = {}
    ppp_debug = bool(getattr(ctx, "config", {}).get("ppp_debug", False)) if hasattr(ctx, "config") else False
    debug_info = []

    for label in registry.labels:
        model = registry.model(label)
        try:
            df_lab, diag = model.simulate_catalog(
                ctx=ctx,
                ra_low=ra_low,
                ra_high=ra_high,
                dec_low=dec_low,
                dec_high=dec_high,
                wave_min=wave_min,
                wave_max=wave_max,
                flux_err=flux_err,
                f_lim=f_lim,
                fake_rate_per_sr_per_A=fake_rate_per_sr_per_A,
                rng=rng,
                nz=nz,
                snr_min=snr_min,
            )
        except TypeError:
            # Backward compatibility: older signature without rng
            df_lab, diag = model.simulate_catalog(
                ctx,
                ra_low,
                ra_high,
                dec_low,
                dec_high,
                wave_min,
                wave_max,
                flux_err,
                f_lim,
                fake_rate_per_sr_per_A,
                seed,
                nz,
                snr_min,
            )
        except Exception as e:
            log(f"[jlc.simulate] Warning: simulate_catalog failed for label={label}: {e}")
            continue
        if df_lab is not None and not df_lab.empty:
            dfs.append(df_lab)
        # collect diagnostics
        if isinstance(diag, dict):
            mu = float(diag.get("mu", 0.0))
            exp_counts[label] = mu
            exp_counts["total"] = exp_counts.get("total", 0.0) + mu
            if "V_Mpc3" in diag:
                label_volumes[label] = float(diag["V_Mpc3"])
            if ppp_debug:
                debug_info.append({"label": label, **{k: v for k, v in diag.items() if k != "label"}})

    # Build final DataFrame
    if len(dfs) == 0:
        df_final = pd.DataFrame(columns=["ra", "dec", "true_class", "wave_obs", "flux_true", "flux_hat", "flux_err"]).copy()
    else:
        import numpy as _np
        df_final = pd.concat(dfs, ignore_index=True)
        # Shuffle rows to mix labels
        if len(df_final) > 1:
            df_final = df_final.sample(frac=1.0, random_state=rng.integers(0, 2**32 - 1)).reset_index(drop=True)

    try:
        lae_mu = exp_counts.get("lae", 0.0)
        oii_mu = exp_counts.get("oii", 0.0)
        fake_mu = exp_counts.get("fake", 0.0)
        lae_V = label_volumes.get("lae", float("nan"))
        oii_V = label_volumes.get("oii", float("nan"))
        total_mu = exp_counts.get("total", 0.0)
        log(f"[jlc.simulate] Expected counts (incl. selection for LAE/OII; fakes from ρ over band): lae≈{lae_mu:.3e}, oii≈{oii_mu:.3e}, fake≈{fake_mu:.3e}; total≈{total_mu:.3e}")
        log(f"[jlc.simulate] Volumes (Mpc^3): lae≈{lae_V:.3e}, oii≈{oii_V:.3e}")
    except Exception:
        pass

    # Compute S/N and apply S/N filter if requested
    if not df_final.empty:
        import numpy as _np
        with _np.errstate(divide='ignore', invalid='ignore'):
            snr = _np.where(df_final["flux_err"].values > 0, df_final["flux_hat"].values / df_final["flux_err"].values, 0.0)
        df_final["snr"] = snr
        if snr_min is not None and np.isfinite(snr_min) and snr_min > 0:
            before = len(df_final)
            df_final = df_final.loc[df_final["snr"] >= float(snr_min)].reset_index(drop=True)
            after = len(df_final)
            try:
                log(f"[jlc.simulate] Applied S/N cut: snr_min={snr_min:.2f}. Kept {after}/{before} rows ({(after/max(before,1))*100:.1f}%).")
            except Exception:
                pass

    return df_final
