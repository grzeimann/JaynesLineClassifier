import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple


LAE_REST = 1215.67  # Angstrom
OII_REST = 3727.0   # Angstrom


@dataclass
class SkyBox:
    ra_low: float
    ra_high: float
    dec_low: float
    dec_high: float

    def sample(self, n: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        ra = rng.uniform(self.ra_low, self.ra_high, size=n)
        dec = rng.uniform(self.dec_low, self.dec_high, size=n)
        return ra, dec


def plot_distributions(df: pd.DataFrame, prefix: str) -> None:
    """Save simple histograms of wavelength and flux by true class.

    Uses common bin grids across classes:
    - wave_obs: linear bins 3500–5500 Å with 40 bins (shared across classes)
    - flux_hat: log-spaced bins 1e-18–1e-15 with 31 bins (shared across classes)
    """
    import matplotlib.pyplot as plt

    classes = sorted(df["true_class"].unique()) if "true_class" in df.columns else ["all"]

    # Define common bin edges
    wave_bins = np.linspace(3500.0, 5500.0, 41)
    flux_bins = np.logspace(np.log10(1e-18), np.log10(1e-15), 32)

    # Wavelength distributions (common linear bins)
    plt.figure(figsize=(6, 4))
    for k in classes:
        sub = df if k == "all" else df[df.true_class == k]
        plt.hist(sub["wave_obs"], bins=wave_bins, alpha=0.6, label=k, histtype="stepfilled")
    plt.xlim(3500.0, 5500.0)
    plt.xlabel("Observed wavelength [A]")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{prefix}_wave.png", dpi=150)
    plt.close()

    # Flux distributions (common log-spaced bins)
    plt.figure(figsize=(6, 4))
    for k in classes:
        sub = df if k == "all" else df[df.true_class == k]
        plt.hist(sub["flux_hat"], bins=flux_bins, alpha=0.6, label=k, histtype="stepfilled", log=True)
    plt.xscale("log")
    plt.xlim(1e-18, 1e-15)
    plt.xlabel("Measured line flux")
    plt.ylabel("Count (log)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{prefix}_flux.png", dpi=150)
    plt.close()


def plot_selection_completeness(selection_model, prefix: str) -> None:
    """Plot 2D completeness images C(F, λ) per label panel using S/N completeness.

    - Wavelength grid: 3500–5500 Å, 40 linear bins (41 edges). Image x-axis uses edges; evaluate at centers.
    - Flux grid: 1e-18–1e-15, 31 log bins (32 edges). Image y-axis log-scaled; evaluate at centers.
    Saves to f"{prefix}_selection.png" as a multi-panel figure (one panel per label in selection_model).
    """
    import matplotlib.pyplot as plt

    # Determine label list from selection model (per-label SN models); fallback to ['all']
    try:
        labels = list(getattr(selection_model, "_sn_models", {}).keys())
        if len(labels) == 0:
            labels = ["all"]
    except Exception:
        labels = ["all"]

    # Shared bin edges and centers (match plot_distributions)
    wave_edges = np.linspace(3500.0, 5500.0, 41)
    flux_edges = np.logspace(np.log10(1e-18), np.log10(1e-15), 32)
    wave_centers = 0.5 * (wave_edges[:-1] + wave_edges[1:])
    # For log bins, geometric mean is a better center
    flux_centers = np.sqrt(flux_edges[:-1] * flux_edges[1:])

    # Build figure with one panel per label
    nlab = len(labels)
    fig, axes = plt.subplots(1, nlab, figsize=(6.0*nlab, 4.8), squeeze=False)
    axes = axes[0]
    for ax, lname in zip(axes, labels):
        nw = wave_centers.size
        nF = flux_centers.size
        C = np.zeros((nw, nF), dtype=float)
        for i, lam in enumerate(wave_centers):
            C[i, :] = np.clip(selection_model.completeness_sn_array(lname, flux_centers, float(lam)), 0.0, 1.0)
        pcm = ax.pcolormesh(wave_edges, flux_edges, C.T, cmap="viridis", shading="auto", vmin=0.0, vmax=1.0)
        ax.set_yscale("log")
        ax.set_xlim(3500.0, 5500.0)
        ax.set_ylim(1e-18, 1e-15)
        ax.set_xlabel("Observed wavelength [A]")
        ax.set_ylabel("Flux")
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


def plot_label_distribution_comparison(df_input: pd.DataFrame, df_inferred: pd.DataFrame, prefix: str, labels: list[str] | None = None, use_hard: bool = False) -> None:
    """Compare input vs posterior-weighted flux–wavelength distributions per label.

    Layout: N columns (labels) × 3 rows
      - Row 1 (top): input true-class 2D histogram H_in (flux vs wavelength)
      - Row 2 (middle): probability-weighted 2D histogram H_pw using p_<label>
      - Row 3 (bottom): residual significance (H_pw − H_in) / sqrt(max(H_in, 1))

    Notes
    - Uses the same (λ,F) grids as plot_selection_completeness for visual parity.
    - Expects df_input to contain columns: wave_obs, flux_hat, true_class.
    - Expects df_inferred to contain columns: wave_obs, flux_hat, and p_<label> for each label.
    - If use_hard is True, middle row uses hard class assignments from argmax across available p_<label>.
    """
    import matplotlib.pyplot as plt

    if df_input is None or df_inferred is None or df_input.empty or df_inferred.empty:
        return

    # Shared bin edges
    wave_edges = np.linspace(3500.0, 5500.0, 41)
    flux_edges = np.logspace(np.log10(1e-18), np.log10(1e-15), 32)

    # Determine labels
    if labels is None:
        # Prefer those appearing in true_class, intersect with available probabilities
        labs_true = sorted(list({str(x) for x in df_input.get("true_class", pd.Series([])).unique() if isinstance(x, (str,))}))
        labs_prob = []
        for col in df_inferred.columns:
            if col.startswith("p_"):
                labs_prob.append(col[2:])
        if len(labs_true) == 0:
            labels = sorted(labs_prob) if len(labs_prob) > 0 else []
        else:
            labels = [L for L in labs_true if f"p_{L}" in df_inferred.columns] or sorted(labs_prob)
    if len(labels) == 0:
        return

    # Prepare hard labels if requested
    hard_idx = None
    hard_names = None
    if use_hard:
        p_cols = [f"p_{L}" for L in labels if f"p_{L}" in df_inferred.columns]
        if len(p_cols) > 0:
            P = df_inferred[p_cols].values
            hard_idx = np.argmax(P, axis=1)
            hard_names = [labels[i] for i in hard_idx]

    # Extract arrays (fallback to df_input for wave/flux if missing in inferred)
    wave_all = (df_inferred["wave_obs"].values if "wave_obs" in df_inferred.columns else df_input["wave_obs"].values).astype(float)
    flux_all = (df_inferred["flux_hat"].values if "flux_hat" in df_inferred.columns else df_input["flux_hat"].values).astype(float)

    nlab = len(labels)
    fig, axes = plt.subplots(3, nlab, figsize=(5.8*nlab, 11.0), squeeze=False)

    for j, L in enumerate(labels):
        # Top: input histogram for true_class==L
        mask_in = (df_input.get("true_class") == L) if "true_class" in df_input.columns else np.zeros(len(df_input), dtype=bool)
        wave_in = df_input.loc[mask_in, "wave_obs"].values.astype(float) if np.any(mask_in) else np.array([])
        flux_in = df_input.loc[mask_in, "flux_hat"].values.astype(float) if np.any(mask_in) else np.array([])
        H_in, xedges, yedges = np.histogram2d(
            wave_in, flux_in, bins=[wave_edges, flux_edges]
        )

        # Middle: probability-weighted histogram
        if use_hard and hard_names is not None:
            w = (np.array(hard_names) == L).astype(float)
        else:
            col = f"p_{L}"
            w = df_inferred[col].values.astype(float) if col in df_inferred.columns else np.zeros_like(wave_all)
        H_pw, _, _ = np.histogram2d(wave_all, flux_all, bins=[wave_edges, flux_edges], weights=w)

        # Bottom: residual significance
        sigma = np.sqrt(np.maximum(H_in, 1.0) + np.maximum(H_pw, 1.0))
        R = (H_pw - H_in) / sigma
        # Clip extreme residuals for visualization
        R = np.clip(R, -5.0, 5.0)

        # Plotting for this label column
        ax_top = axes[0, j]
        ax_mid = axes[1, j]
        ax_bot = axes[2, j]

        pcm1 = ax_top.pcolormesh(xedges, yedges, H_in.T, cmap="magma", shading="auto")
        ax_top.set_title(f"{L}: input true")
        ax_top.set_xlim(wave_edges[0], wave_edges[-1])
        ax_top.set_ylim(flux_edges[0], flux_edges[-1])
        ax_top.set_yscale("log")
        ax_top.set_ylabel("Flux")
        fig.colorbar(pcm1, ax=ax_top, fraction=0.046, pad=0.04, label="count")

        pcm2 = ax_mid.pcolormesh(xedges, yedges, H_pw.T, cmap="magma", shading="auto")
        ax_mid.set_title(f"{L}: posterior-weighted")
        ax_mid.set_xlim(wave_edges[0], wave_edges[-1])
        ax_mid.set_ylim(flux_edges[0], flux_edges[-1])
        ax_mid.set_yscale("log")
        ax_mid.set_ylabel("Flux")
        fig.colorbar(pcm2, ax=ax_mid, fraction=0.046, pad=0.04, label="weighted count")

        pcm3 = ax_bot.pcolormesh(xedges, yedges, R.T, cmap="coolwarm", shading="auto", vmin=-5.0, vmax=5.0)
        ax_bot.set_title(f"{L}: (weighted - input)/sqrt(input)")
        ax_bot.set_xlim(wave_edges[0], wave_edges[-1])
        ax_bot.set_ylim(flux_edges[0], flux_edges[-1])
        ax_bot.set_yscale("log")
        ax_bot.set_xlabel("Observed wavelength [A]")
        ax_bot.set_ylabel("Flux")
        fig.colorbar(pcm3, ax=ax_bot, fraction=0.046, pad=0.04, label="σ")

    fig.tight_layout()
    out_path = f"{prefix}_compare.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    try:
        from jlc.utils.logging import log as _log
        _log(f"[jlc.simulate] Wrote input vs posterior-weighted distribution comparison to {out_path}")
    except Exception:
        pass


def plot_probability_circle(df_inferred: pd.DataFrame, prefix: str, labels: list[str] | None = None, title: str | None = None, alpha: float = 0.5, size: float = 12.0, hexbin_cmap: str | None = "viridis") -> None:
    """Plot posterior label probabilities as points in a unit-circle embedding.

    - Each label is placed on the unit circle at angle theta_k = 2π k / N.
    - Each row is mapped to a 2D point x = sum_k p_k v_k, where v_k is the
      unit vector of label k.
    - Points near the circle indicate confident posteriors; near the center
      indicate ambiguous posteriors.

    Inputs
    - df_inferred: DataFrame containing posterior columns p_<label> per label.
    - prefix: output path prefix; image will be saved to f"{prefix}_circle.png".
    - labels: optional list of label names in the same order as p_ columns. If
      None, inferred from df_inferred columns beginning with 'p_'.
    - title: optional plot title.
    - alpha: point transparency for scatter.
    - size: scatter marker size.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    if df_inferred is None or df_inferred.empty:
        return

    # Determine labels from posterior columns if not provided
    if labels is None:
        labels = [c[2:] for c in df_inferred.columns if isinstance(c, str) and c.startswith("p_")]
        labels = sorted(labels)
    # Filter to labels that actually exist in columns
    labels = [L for L in (labels or []) if f"p_{L}" in df_inferred.columns]
    N = len(labels)
    if N == 0:
        return

    # Build probability matrix in chosen order
    P = df_inferred[[f"p_{L}" for L in labels]].values.astype(float)
    # Normalize rows defensively to sum to 1
    row_sums = P.sum(axis=1, keepdims=True)
    # Avoid division by zero: if a row sums to 0, leave it as zeros (maps to origin)
    safe = np.where(row_sums > 0, row_sums, 1.0)
    P = P / safe

    # Unit-circle label vectors
    angles = 2 * np.pi * np.arange(N) / max(N, 1)
    label_x = np.cos(angles)
    label_y = np.sin(angles)
    V = np.stack((label_x, label_y), axis=1)  # (N,2)

    # Map probabilities to points
    XY = P @ V  # (n,2)
    x_points = XY[:, 0]
    y_points = XY[:, 1]

    r = np.sqrt(x_points ** 2 + y_points ** 2)
    theta_points = np.arctan2(y_points, x_points)  # [-π, π]
    theta_norm = (theta_points + np.pi) / (2 * np.pi)  # [0, 1]
    label_idx = np.argmax(P, axis=1)
    cmap = plt.get_cmap("tab10")
    colors = cmap(label_idx % 10)
    #colors = plt.get_cmap("coolwarm")(theta_norm)
    sizes = 3 + 20 * r
    alphas = 0.05 + 0.95 * r

    # Plot
    fig, ax = plt.subplots(figsize=(6, 6))

    # Reference circle and certainty rings
    theta = np.linspace(0, 2 * np.pi, 512)
    ax.plot(np.cos(theta), np.sin(theta), linestyle="--", linewidth=1, alpha=0.3)
    for r in (0.3, 0.6, 0.9):
        ax.plot(r * np.cos(theta), r * np.sin(theta), linewidth=0.5, alpha=0.2)

    # Scatter points
    #ax.scatter(x_points, y_points, color=colors, s=sizes, alpha=alphas)

    # --- density plot ---
    hb = ax.hexbin(
        x_points,
        y_points,
        gridsize=50,  # adjust resolution
        extent=[-1.2, 1.2, -1.2, 1.2],
        mincnt=1,  # ignore empty bins
        cmap=hexbin_cmap,
        norm=LogNorm()  # log color scale, good for wide ranges
    )
    # Add colorbar for hexbin counts
    cbar = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("count per hex (log scale)")
    
    # Label anchors on circle and text
    ax.scatter(label_x, label_y, s=10, marker="o", facecolor='grey', edgecolor="black", zorder=3)
    for x, y, lab in zip(label_x, label_y, labels):
        ax.text(1.08 * x, 1.08 * y, lab, ha="center", va="center", fontsize=10, fontweight="bold")

    # Aesthetics
    ax.set_aspect("equal", "box")
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.25, 1.25)
    ax.set_xticks([])
    ax.set_yticks([])
    if title is None:
        title = "Posterior label probabilities (unit-circle embedding)"
    ax.set_title(title, pad=12)
    ax.grid(alpha=0.1)

    import os as _os
    _os.makedirs(_os.path.dirname(f"{prefix}_circle.png") or ".", exist_ok=True)
    out_path = f"{prefix}_circle.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    try:
        from jlc.utils.logging import log as _log
        _log(f"[jlc.simulate] Wrote probability circle plot to {out_path}")
    except Exception:
        pass
