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
    """Plot a 2D image of completeness C(F, λ) over the common flux/wave grid.

    - Wavelength grid: 3500–5500 Å, 40 linear bins (41 edges). Image x-axis uses edges; evaluate at centers.
    - Flux grid: 1e-18–1e-15, 31 log bins (32 edges). Image y-axis log-scaled; evaluate at centers.
    Saves to f"{prefix}_selection.png".
    """
    import matplotlib.pyplot as plt

    # Shared bin edges and centers (match plot_distributions)
    wave_edges = np.linspace(3500.0, 5500.0, 41)
    flux_edges = np.logspace(np.log10(1e-18), np.log10(1e-15), 32)
    wave_centers = 0.5 * (wave_edges[:-1] + wave_edges[1:])
    # For log bins, geometric mean is a better center
    flux_centers = np.sqrt(flux_edges[:-1] * flux_edges[1:])

    # Build completeness grid: shape (n_wave, n_flux)
    nw = wave_centers.size
    nF = flux_centers.size
    C = np.zeros((nw, nF), dtype=float)
    for i, lam in enumerate(wave_centers):
        C[i, :] = np.clip(selection_model.completeness(flux_centers, float(lam)), 0.0, 1.0)

    # Plot using pcolormesh with edges
    plt.figure(figsize=(6.5, 4.8))
    # pcolormesh expects array shaped (ny, nx) for Z when Y,X are 1D edges
    pcm = plt.pcolormesh(wave_edges, flux_edges, C.T, cmap="viridis", shading="auto", vmin=0.0, vmax=1.0)
    plt.yscale("log")
    plt.xlim(3500.0, 5500.0)
    plt.ylim(1e-18, 1e-15)
    plt.xlabel("Observed wavelength [A]")
    plt.ylabel("Flux")
    cbar = plt.colorbar(pcm)
    cbar.set_label("Selection completeness C(F, λ)")
    plt.tight_layout()
    out_path = f"{prefix}_selection.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    try:
        from jlc.utils.logging import log as _log
        _log(f"[jlc.simulate] Wrote selection completeness image to {out_path}")
    except Exception:
        pass
