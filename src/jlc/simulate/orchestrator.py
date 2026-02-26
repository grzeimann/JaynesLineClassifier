from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Sequence, Tuple, Optional
import numpy as np

from .pipeline import NoiseCubeReader, LambdaSliceSpaxelIndex, wcs_from_indices, build_catalog_table
from .noise_histogram import NoiseHistogram
from .completeness_providers import NoiseBinConditionedCompleteness
from .rate_integrand import rate_density_integrand_per_flux


@dataclass
class KernelEnv:
    """Minimal noise environment placeholder for kernel stubs."""
    lam: float
    noise: float


def draw_signal_and_flux_stub(F_true: float, lam: float, noise_env: KernelEnv, rng: np.random.Generator) -> Tuple[float, float, float]:
    """Minimal kernel stub: returns (signal, F_fit, sigma_F).

    - signal approximates F_true (placeholder for shape-based signal metric).
    - F_fit = F_true + N(0, sigma_F) with sigma_F = noise_env.noise.
    - sigma_F equals the noise value from the cell.
    """
    sigma_F = float(max(0.0, noise_env.noise))
    if not np.isfinite(sigma_F) or sigma_F <= 0:
        sigma_F = 0.0
    if sigma_F > 0:
        F_fit = float(F_true) + float(rng.normal(loc=0.0, scale=sigma_F))
    else:
        F_fit = float(F_true)
    signal = float(F_true)  # placeholder
    return signal, F_fit, sigma_F


def expected_counts_per_cell(
    lambda_grid: Sequence[float],
    noise_bin_edges: Sequence[float],
    noise_hist: NoiseHistogram,
    lf_by_label: Dict[str, Any],
    F_true_grid: np.ndarray,
    selection,
    *,
    enable_mem: bool = False,
) -> Dict[str, np.ndarray]:
    """Compute expected counts mu[label][k, j] per (λ, noise-bin) cell.

    Integrates rate_density_integrand_per_flux over F_true and scales by survey area.
    """
    from jlc.utils.logging import log as _log
    import time as _time

    lam_arr = np.asarray(lambda_grid, dtype=float)
    nbins = int(np.asarray(noise_bin_edges, dtype=float).size - 1)
    labels = list(lf_by_label.keys())
    mu: Dict[str, np.ndarray] = {lbl: np.zeros((lam_arr.size, nbins), dtype=float) for lbl in labels}

    # Optional memory diagnostics
    if enable_mem:
        try:
            from .diagnostics import log_mem, array_nbytes
            extra = {f"mu[{lbl}]": array_nbytes(arr) for lbl, arr in mu.items()}
            log_mem("mu_alloc:init", extra)
        except Exception:
            pass

    t0 = _time.time()
    step = max(1, lam_arr.size // 20)

    for k, lam in enumerate(lam_arr):
        centers, weights, _ = noise_hist.hist_at_lambda(float(lam))
        # Use bin centers as representative noise values
        for j, n_val in enumerate(centers):
            if weights[j] <= 0:
                continue
            provider = NoiseBinConditionedCompleteness(selection=selection, noise_value=float(n_val))
            for lbl in labels:
                r_F = rate_density_integrand_per_flux(F_true_grid=F_true_grid, lam=float(lam), label=lbl, lf_params=lf_by_label[lbl], completeness_provider=provider)
                # Integrate over F to get rate per sr·Å (simple trapezoidal)
                rate_per_srA = float(np.trapz(r_F, F_true_grid))
                # Convert to absolute counts using per-λ survey area from histogram
                area_sr = float(noise_hist.survey_area_sr[k])
                mu[lbl][k, j] = rate_per_srA * area_sr
        # Progress log
        if (k % step == 0) or (k == lam_arr.size - 1):
            elapsed = _time.time() - t0
            done = k + 1
            rate = done / elapsed if elapsed > 0 else float('inf')
            remaining = lam_arr.size - done
            eta = remaining / rate if rate > 0 else float('inf')
            pct = (100.0 * done / lam_arr.size) if lam_arr.size > 0 else 100.0
            try:
                partial_totals = {lbl: float(np.nansum(mu[lbl][:done, :])) for lbl in labels}
                parts = ", ".join([f"{k}≈{v:.2f}" for k, v in partial_totals.items()])
                _log(f"[jlc.simulate] Expected-counts: {done}/{lam_arr.size} λ-slices ({pct:.0f}%), elapsed {elapsed:.1f}s, ETA {eta:.1f}s; partial totals: {parts}")
            except Exception:
                _log(f"[jlc.simulate] Expected-counts: {done}/{lam_arr.size} λ-slices ({pct:.0f}%), elapsed {elapsed:.1f}s, ETA {eta:.1f}s")
            if enable_mem:
                try:
                    from .diagnostics import log_mem, array_nbytes
                    extra = {f"mu[{lbl}]": array_nbytes(arr) for lbl, arr in mu.items()}
                    log_mem("mu_alloc:progress", extra)
                except Exception:
                    pass
    return mu


def sample_counts(mu_dict: Dict[str, np.ndarray], rng: np.random.Generator) -> Dict[str, np.ndarray]:
    """Poisson sample counts per cell for each label.

    Returns a dict with same shapes as mu_dict, dtype=int64.
    """
    out: Dict[str, np.ndarray] = {}
    for lbl, mu in mu_dict.items():
        mu_clip = np.where(np.isfinite(mu) & (mu >= 0), mu, 0.0)
        out[lbl] = rng.poisson(mu_clip).astype(int)
    return out


def simulate_sources_for_cell(
    k: int,
    j: int,
    label: str,
    N_kj: int,
    lambda_grid: Sequence[float],
    F_true_grid: np.ndarray,
    lf_model: Any,
    lambda_indexer: LambdaSliceSpaxelIndex,
    noise_bin_center: float,
    rng: np.random.Generator,
    cube_reader: NoiseCubeReader,
    selection: Any | None = None,
    *,
    include_debug: bool = False,
    profile: Any | None = None,
    draw_func: Any | None = None,
) -> list:
    """Simulate N_kj sources in a given (k, j, label) cell using minimal kernel stub.

    Returns a list of record dicts ready for build_catalog_table().
    """
    if N_kj <= 0:
        return []
    lam = float(np.asarray(lambda_grid, dtype=float)[k])
    # 1) RA/Dec via spaxel sampling
    ira, idec = lambda_indexer.sample_spaxels(j, N_kj, rng)
    if ira.size == 0:
        return []
    ra, dec = wcs_from_indices(cube_reader.cube, ira, idec)
    # 2) Sample F_true with weights proportional to the same rate integrand used in Phase 1
    #    This ensures the flux distribution reflects LF→L mapping, cosmology, and completeness
    from .completeness_providers import NoiseBinConditionedCompleteness as _NBCC
    from .rate_integrand import rate_density_integrand_per_flux as _rint
    # Build provider with the current bin's representative noise and the selection model
    provider = _NBCC(selection=selection, noise_value=float(noise_bin_center))
    try:
        rF = _rint(F_true_grid=F_true_grid, lam=lam, label=label, lf_params=lf_model, completeness_provider=provider)
    except Exception:
        # Conservative fallback: uniform in log F
        rF = np.ones_like(F_true_grid, dtype=float)
    rF = np.where(np.isfinite(rF) & (rF >= 0), rF, 0.0)
    s = float(rF.sum())
    if not np.isfinite(s) or s <= 0:
        w = np.ones_like(F_true_grid, dtype=float) / float(max(F_true_grid.size, 1))
    else:
        w = rF / s
    cdf = np.cumsum(w)
    if cdf.size > 0:
        cdf[-1] = 1.0
    u = rng.random(size=int(N_kj))
    idx = np.searchsorted(cdf, u, side="right")
    idx = np.clip(idx, 0, F_true_grid.size - 1)
    F_true_samples = F_true_grid[idx]
    # 3) Kernel to generate signal, F_fit, F_error (prefer provided draw_func/profile; fallback to stub)
    env = KernelEnv(lam=lam, noise=float(noise_bin_center))
    # Resolve draw function
    _draw = draw_func
    if _draw is None:
        try:
            from .kernel import draw_signal_and_flux as _kernel_draw
            _draw = _kernel_draw
        except Exception:
            _draw = None
    records = []
    for i in range(int(N_kj)):
        F_true = float(F_true_samples[i])
        if _draw is not None:
            try:
                signal, F_fit, sigma_F = _draw(F_true=F_true, lam=lam, noise_env=env, rng=rng, profile=profile)
            except Exception:
                signal, F_fit, sigma_F = draw_signal_and_flux_stub(F_true=F_true, lam=lam, noise_env=env, rng=rng)
        else:
            signal, F_fit, sigma_F = draw_signal_and_flux_stub(F_true=F_true, lam=lam, noise_env=env, rng=rng)
        rec = {
            "ra": float(ra[i]),
            "dec": float(dec[i]),
            "lambda": lam,
            "F_true": F_true,
            "F_fit": float(F_fit),
            "F_error": float(sigma_F),
            "signal": float(signal),
            "noise": float(noise_bin_center),
            "label": str(label),
        }
        if include_debug:
            rec.update({
                "ira": int(ira[i]),
                "idec": int(idec[i]),
                "k": int(k),
                "j": int(j),
            })
        records.append(rec)
    return records


def run_simulation(
    cube_reader: NoiseCubeReader,
    noise_hist: NoiseHistogram,
    noise_bin_edges: Sequence[float],
    lf_by_label: Dict[str, Any],
    F_true_grid: np.ndarray,
    selection,
    rng: Optional[np.random.Generator] = None,
    *,
    enable_mem: bool = False,
    include_debug: bool = False,
    profile_by_label: Optional[Dict[str, Any]] = None,
    draw_func: Any | None = None,
) -> np.recarray:
    """High-level simulation driver combining histogram, completeness, LF, and spatial placement.

    Returns a numpy recarray via build_catalog_table().
    """
    from jlc.utils.logging import log as _log
    import time as _time

    if rng is None:
        rng = np.random.default_rng()
    lambda_grid = cube_reader.cube.wave_grid

    t_phase0 = _time.time()
    # 1) Expected counts per cell
    _log("[jlc.simulate] Phase 1/3: computing expected counts per (λ, noise-bin) cell…")
    mu = expected_counts_per_cell(lambda_grid=lambda_grid, noise_bin_edges=noise_bin_edges, noise_hist=noise_hist, lf_by_label=lf_by_label, F_true_grid=F_true_grid, selection=selection, enable_mem=enable_mem)
    t_phase1 = _time.time()
    _log(f"[jlc.simulate] Phase 1/3 done in {t_phase1 - t_phase0:.2f}s")
    if enable_mem:
        try:
            from .diagnostics import log_mem, array_nbytes
            extra = {f"mu[{lbl}]": array_nbytes(arr) for lbl, arr in mu.items()}
            log_mem("phase1:mu_final", extra)
        except Exception:
            pass

    # Diagnostics: log total expected counts per label and overall
    try:
        totals = {lbl: float(np.nansum(arr)) for lbl, arr in mu.items()}
        total_all = float(sum(totals.values()))
        parts = ", ".join([f"{k}≈{v:.2f}" for k, v in totals.items()])
        _log(f"[jlc.simulate] Experimental expected total counts per label: {parts}; overall≈{total_all:.2f}")
    except Exception:
        pass
    # Completeness summary (if tracing enabled)
    try:
        from .diagnostics import get_completeness_tracer as _gct
        tracer = _gct()
        tracer.summarize()
    except Exception:
        pass

    # 2) Sample counts
    _log("[jlc.simulate] Phase 2/3: Poisson sampling counts per cell…")
    if enable_mem:
        try:
            from .diagnostics import log_mem
            log_mem("phase2:before_poisson")
        except Exception:
            pass
    N = sample_counts(mu, rng)
    if enable_mem:
        try:
            from .diagnostics import log_mem, array_nbytes
            extra = {f"N[{lbl}]": array_nbytes(arr) for lbl, arr in N.items()}
            log_mem("phase2:after_poisson", extra)
        except Exception:
            pass
    t_phase2 = _time.time()
    _log(f"[jlc.simulate] Phase 2/3 done in {t_phase2 - t_phase1:.2f}s")

    # 3) Generate records
    _log("[jlc.simulate] Phase 3/3: generating catalog records…")
    records = []
    step = max(1, int(lambda_grid.size) // 20)
    t_gen0 = _time.time()
    for k, lam in enumerate(lambda_grid):
        noise_slice = cube_reader.read_noise_slice(k)
        indexer = LambdaSliceSpaxelIndex(noise_slice_2d=noise_slice, noise_bin_edges=noise_bin_edges)
        centers = 0.5 * (np.asarray(noise_bin_edges, dtype=float)[:-1] + np.asarray(noise_bin_edges, dtype=float)[1:])
        for j, n_val in enumerate(centers):
            for label, N_lbl in N.items():
                N_kj = int(N_lbl[k, j])
                if N_kj <= 0:
                    continue
                # Resolve per-label profile if provided
                prof = None
                try:
                    if profile_by_label is not None:
                        prof = profile_by_label.get(str(label))
                except Exception:
                    prof = None
                recs = simulate_sources_for_cell(
                    k=k,
                    j=j,
                    label=label,
                    N_kj=N_kj,
                    lambda_grid=lambda_grid,
                    F_true_grid=F_true_grid,
                    lf_model=lf_by_label[label],
                    lambda_indexer=indexer,
                    noise_bin_center=float(n_val),
                    rng=rng,
                    cube_reader=cube_reader,
                    selection=selection,
                    include_debug=include_debug,
                    profile=prof,
                    draw_func=draw_func,
                )
                if len(recs) > 0:
                    records.extend(recs)
        # Progress update per λ
        if (k % step == 0) or (k == lambda_grid.size - 1):
            done = k + 1
            elapsed = _time.time() - t_gen0
            rate = (done / elapsed) if elapsed > 0 else float('inf')
            remain = lambda_grid.size - done
            eta = remain / rate if rate > 0 else float('inf')
            pct = (100.0 * done / lambda_grid.size) if lambda_grid.size > 0 else 100.0
            _log(f"[jlc.simulate] Record gen: {done}/{lambda_grid.size} λ-slices ({pct:.0f}%), records so far {len(records)}, elapsed {elapsed:.1f}s, ETA {eta:.1f}s")

    t_phase3 = _time.time()
    _log(f"[jlc.simulate] Phase 3/3 done in {t_phase3 - t_phase2:.2f}s; total records {len(records)}")
    _log(f"[jlc.simulate] Experimental simulation total time {t_phase3 - t_phase0:.2f}s")

    return build_catalog_table(records)
