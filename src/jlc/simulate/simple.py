import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, Dict, Optional


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


def _sample_flux_from_powerlaw(n: int, f_min: float, f_max: float, alpha: float, rng: np.random.Generator) -> np.ndarray:
    """Sample positive fluxes from a simple power-law N(F) ~ F^alpha between [f_min, f_max].
    This is a fast placeholder; for realism plug in LF-based sampling later.
    alpha != -1 assumed; if near -1 we jitter slightly.
    """
    f_min = max(float(f_min), 1e-30)
    f_max = max(float(f_max), f_min * 1.0001)
    a = float(alpha)
    if abs(a + 1.0) < 1e-6:
        a = -0.999  # avoid division by zero in inverse CDF
    u = rng.random(n)
    c1 = f_max ** (a + 1.0)
    c0 = f_min ** (a + 1.0)
    return ((u * (c1 - c0) + c0)) ** (1.0 / (a + 1.0))


def simulate_catalog(
    n: int,
    sky: SkyBox,
    f_lim: float,
    class_fracs: Dict[str, float],
    wave_min: float,
    wave_max: float,
    flux_err: float = 1e-17,
    seed: Optional[int] = None,
    snr_min: Optional[float] = None,
) -> pd.DataFrame:
    """Generate a simple mock catalog with true_class, RA/DEC, observed wavelength and measured flux.

    - Sky is uniform in a rectangular box.
    - Selection is a hard cut: retain only objects with flux_hat > f_lim.
    - Classes: 'lae', 'oii', 'fake'.
    - Wavelengths for line classes are drawn from a uniform redshift mapped via rest wavelengths
      but we directly draw observed wavelengths uniformly within [wave_min, wave_max] for simplicity.
    - Fluxes: line classes from a power-law, fakes from a log-normal below threshold.
    """
    rng = np.random.default_rng(seed)

    # Normalize class fractions
    keys = ["lae", "oii", "fake"]
    w = np.array([max(class_fracs.get(k, 0.0), 0.0) for k in keys], dtype=float)
    if w.sum() <= 0:
        w = np.array([1.0, 1.0, 1.0])
    w = w / w.sum()

    # Assign classes
    cls_idx = rng.choice(len(keys), size=n, p=w)
    cls = np.array([keys[i] for i in cls_idx])

    # Sky positions
    ra, dec = sky.sample(n, rng)

    # Observed wavelength uniform in band
    wave_obs = rng.uniform(wave_min, wave_max, size=n)

    # True flux sampling by class
    F_true = np.zeros(n, dtype=float)
    # Line-emitter fluxes: heavy tail power-law with slope -1.5
    mask_lae = cls == "lae"
    mask_oii = cls == "oii"
    n_lae = int(mask_lae.sum())
    n_oii = int(mask_oii.sum())
    if n_lae > 0:
        F_true[mask_lae] = _sample_flux_from_powerlaw(n_lae, 1e-20, 5e-16, -1.5, rng)
    if n_oii > 0:
        F_true[mask_oii] = _sample_flux_from_powerlaw(n_oii, 1e-20, 3e-16, -1.3, rng)

    # Fakes: log-normal mostly below threshold
    mask_fake = cls == "fake"
    n_fake = int(mask_fake.sum())
    if n_fake > 0:
        mu = np.log(max(f_lim, 1e-20)) - 2.0  # mean well below threshold in log-space
        sigma_ln = 1.0
        F_true[mask_fake] = rng.lognormal(mean=mu, sigma=sigma_ln, size=n_fake)

    # Measurements: add Gaussian noise
    flux_err_arr = np.full(n, float(flux_err), dtype=float)
    flux_hat = F_true + rng.normal(0.0, flux_err_arr)
    flux_hat = np.clip(flux_hat, a_min=0.0, a_max=None)  # non-negative measured flux

    # Compute S/N and apply selection: flux_hat > f_lim and (S/N >= snr_min if provided)
    with np.errstate(divide='ignore', invalid='ignore'):
        snr = np.where(flux_err_arr > 0, flux_hat / flux_err_arr, 0.0)
    sel = flux_hat > f_lim
    if snr_min is not None and np.isfinite(snr_min) and snr_min > 0:
        sel = sel & (snr >= float(snr_min))

    df = pd.DataFrame({
        "ra": ra,
        "dec": dec,
        "true_class": cls,
        "wave_obs": wave_obs,
        "flux_hat": flux_hat,
        "flux_err": flux_err_arr,
        "snr": snr,
    })

    return df.loc[sel].reset_index(drop=True)


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
