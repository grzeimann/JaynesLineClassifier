from __future__ import annotations
import os
from typing import Optional, List
import numpy as np
import pandas as pd

from .simple import (
    plot_distributions,
    plot_label_distribution_comparison,
    plot_probability_circle,
)
from .completeness_providers import NoiseHistogramCompleteness


def _write_selection_plot(prefix: str, selection: object, noise_hist: object, F_true_grid: np.ndarray) -> None:
    """Write completeness image using NoiseHistogramCompleteness over (λ, F_true).

    - λ grid: use noise_hist.lambda_grid
    - F grid: use provided F_true_grid (assumed positive, log-spaced recommended)
    - One panel per label present in selection._sn_models; if none, single 'all' panel
    """
    import matplotlib.pyplot as plt
    # Determine labels from selection
    try:
        labels = list(getattr(selection, "_sn_models", {}).keys())
        if len(labels) == 0:
            labels = ["all"]
    except Exception:
        labels = ["all"]

    lam_grid = np.asarray(getattr(noise_hist, "lambda_grid", np.array([])), dtype=float)
    if lam_grid.size == 0 or F_true_grid is None or F_true_grid.size == 0:
        return

    # Build provider that marginalizes over noise histogram per λ
    prov = NoiseHistogramCompleteness(selection=selection, hist=noise_hist)

    # Prepare image grids
    wave_edges = lam_grid  # treat as edges if monotonic; for pcolormesh need edges length = centers+1
    # If lambda grid is uniform, we can synthesize edges by midpoints
    if lam_grid.size >= 2:
        centers = lam_grid
        mids = 0.5 * (centers[:-1] + centers[1:])
        wave_edges = np.empty(centers.size + 1, dtype=float)
        wave_edges[1:-1] = mids
        # extrapolate edges
        wave_edges[0] = centers[0] - (mids[0] - centers[0])
        wave_edges[-1] = centers[-1] + (centers[-1] - mids[-1])
    else:
        wave_edges = np.array([lam_grid[0] - 0.5, lam_grid[0] + 0.5], dtype=float)

    # Flux edges: derive from F_true_grid geometrically if possible
    F = np.asarray(F_true_grid, dtype=float)
    # Ensure positive
    F = F[np.isfinite(F) & (F > 0)]
    if F.size < 2:
        return
    # Construct log-spaced edges around centers
    ratios = F[1:] / F[:-1]
    gm = np.sqrt(ratios)
    F_edges = np.empty(F.size + 1, dtype=float)
    F_edges[1:-1] = F[:-1] * gm
    # Extrapolate first and last using same ratio as nearest interior
    r0 = ratios[0]
    r1 = ratios[-1]
    F_edges[0] = F[0] / max(r0, 1.0000001)
    F_edges[-1] = F[-1] * max(r1, 1.0000001)

    # Compute completeness per label
    nlam = lam_grid.size
    nF = F.size
    nlab = len(labels)
    fig, axes = plt.subplots(1, nlab, figsize=(6.0 * nlab, 4.8), squeeze=False)
    axes = axes[0]
    for ax, lname in zip(axes, labels):
        C = np.zeros((nlam, nF), dtype=float)
        for i, lam in enumerate(lam_grid):
            try:
                C[i, :] = np.clip(prov.completeness(F, float(lam), str(lname)), 0.0, 1.0)
            except Exception:
                C[i, :] = 0.0
        pcm = ax.pcolormesh(wave_edges, F_edges, C.T, cmap="viridis", shading="auto", vmin=0.0, vmax=1.0)
        ax.set_yscale("log")
        ax.set_xlabel("Observed wavelength [A]")
        ax.set_ylabel("Flux (F_true)")
        ax.set_title(str(lname))
        cbar = fig.colorbar(pcm, ax=ax)
        cbar.set_label("C(F, λ)")
    fig.tight_layout()
    out_path = f"{prefix}_selection.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    try:
        from jlc.utils.logging import log as _log
        _log(f"[jlc.simulate] Wrote selection completeness image to {out_path}")
    except Exception:
        pass


def _ensure_dir_for_prefix(prefix: str) -> None:
    d = os.path.dirname(prefix)
    if d:
        os.makedirs(d, exist_ok=True)


def _to_default_schema(df_sim: pd.DataFrame) -> pd.DataFrame:
    """Map experimental sim catalog columns to the default plotting schema.

    Input df_sim columns expected at least: ['lambda', 'F_fit', 'F_error', 'label']
    Output columns for plotting helpers:
      - wave_obs
      - flux_hat
      - flux_err
      - true_class
    """
    df = pd.DataFrame()
    if df_sim is None or len(df_sim) == 0:
        return df
    # Defensive casting and renaming
    df["wave_obs"] = df_sim.get("lambda", np.array([], dtype=float)).astype(float, copy=False)
    df["flux_hat"] = df_sim.get("F_fit", np.array([], dtype=float)).astype(float, copy=False)
    df["flux_err"] = df_sim.get("F_error", np.array([], dtype=float)).astype(float, copy=False)
    # true_class mirrors the label column in simulated output
    lab = df_sim.get("label")
    if lab is None:
        df["true_class"] = pd.Series(["unknown"] * len(df))
    else:
        df["true_class"] = lab.astype(str)
    return df


def _posterior_like_df(df_input: pd.DataFrame, labels: Optional[List[str]] = None) -> pd.DataFrame:
    """Synthesize a posterior DataFrame with one-hot posteriors along true_class.

    This lets us reuse comparison and circle plots. If labels is None, infer from
    unique true_class values.
    """
    if df_input is None or df_input.empty:
        return pd.DataFrame()
    if labels is None:
        labels = sorted([str(x) for x in df_input["true_class"].unique() if isinstance(x, (str,))])
    out = pd.DataFrame()
    out["wave_obs"] = df_input["wave_obs"].values.astype(float)
    out["flux_hat"] = df_input["flux_hat"].values.astype(float)
    # Build one-hot
    for L in labels:
        out[f"p_{L}"] = (df_input["true_class"].values == L).astype(float)
    return out


def _plot_per_label_flux_histograms(df_input: pd.DataFrame, prefix: str, labels: Optional[List[str]] = None) -> None:
    """Generate simple flux histograms per label to match sim_lae.png, sim_oii.png style.

    Saves to f"{prefix}_<label>.png" for each label present.
    """
    import matplotlib.pyplot as plt
    if df_input is None or df_input.empty:
        return
    if labels is None:
        labels = sorted([str(x) for x in df_input["true_class"].unique() if isinstance(x, (str,))])
    if len(labels) == 0:
        labels = ["all"]
    bins = np.logspace(np.log10(1e-18), np.log10(1e-14), 50)
    for L in labels:
        sub = df_input if L == "all" else df_input[df_input["true_class"] == L]
        plt.figure(figsize=(6, 4))
        plt.hist(sub["flux_hat"].values.astype(float), bins=bins, histtype="stepfilled", alpha=0.7)
        plt.xscale("log")
        plt.xlabel("Measured line flux")
        plt.ylabel("Count")
        plt.tight_layout()
        out = f"{prefix}_{L}.png"
        plt.savefig(out, dpi=150)
        plt.close()
        try:
            from jlc.utils.logging import log as _log
            _log(f"[jlc.simulate] Wrote per-label flux histogram to {out}")
        except Exception:
            pass


def write_experimental_plots(prefix: str, df_sim: pd.DataFrame, selection: object | None = None, noise_hist: object | None = None, F_true_grid: np.ndarray | None = None) -> None:
    """Write a suite of diagnostic plots for the experimental simulation output.

    Produces files compatible with the default sim mode naming convention:
      - {prefix}_wave.png
      - {prefix}_flux.png
      - {prefix}_lae.png, {prefix}_oii.png (per-label flux hists; per labels present)
      - {prefix}_compare.png (input vs posterior-weighted, using one-hot posteriors)
      - {prefix}_circle.png (posterior embedding; one-hot)
      - {prefix}_selection.png (completeness C(F,λ) per label using NoiseHistogramCompleteness)
    """
    _ensure_dir_for_prefix(prefix)
    df_in = _to_default_schema(df_sim)
    if df_in is None or df_in.empty:
        return
    # Core distributions (wave/flux)
    try:
        plot_distributions(df_in.copy(), prefix)
    except Exception:
        pass
    # Per-label histograms
    try:
        labs = sorted([str(x) for x in df_in["true_class"].unique() if isinstance(x, (str,))])
        _plot_per_label_flux_histograms(df_in, prefix, labs)
    except Exception:
        labs = None
    # Comparison using synthesized posteriors
    try:
        df_post = _posterior_like_df(df_in, labels=labs if labs is not None else None)
        if not df_post.empty:
            plot_label_distribution_comparison(df_input=df_in, df_inferred=df_post, prefix=prefix, labels=labs if labs is not None else None, use_hard=True)
            plot_probability_circle(df_inferred=df_post, prefix=prefix, labels=labs if labs is not None else None)
    except Exception:
        pass
    # Selection completeness plot using NoiseHistogramCompleteness when available
    try:
        if (selection is not None) and (noise_hist is not None) and (F_true_grid is not None) and (F_true_grid.size > 0):
            _write_selection_plot(prefix, selection, noise_hist, F_true_grid)
    except Exception:
        pass
