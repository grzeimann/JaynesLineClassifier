from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Tuple, Optional, Dict, Any
import numpy as np


class LineProfile(Protocol):
    """Protocol for line-shape profile objects used by the kernel.

    A LineProfile maps a latent line with total flux F_true at wavelength lam
    into a scalar detection metric called `signal` that downstream selection or
    measurement code might consume. For this minimal implementation, `signal`
    is a simple deterministic function of F_true and an intrinsic width.
    """

    def signal(self, F_true: float, lam: float) -> float:  # pragma: no cover - interface
        ...


@dataclass
class GaussianLineProfile:
    """Minimal Gaussian line profile with a single width parameter.

    Parameters
    ----------
    sigma_A : float
        Intrinsic line width in Angstrom (not used to broaden anything yet,
        but carried for future kernel growth and tests).
    gain : float
        Linear gain mapping from F_true to the scalar `signal` reported by the
        kernel. Set to 1.0 by default so signal≈F_true in current plots.
    """
    sigma_A: float = 2.0
    gain: float = 1.0

    def signal(self, F_true: float, lam: float) -> float:
        # For the first cut, define a simple proportional mapping.
        try:
            s = float(self.gain) * float(F_true)
            if not np.isfinite(s):
                s = 0.0
            return float(s)
        except Exception:
            return 0.0


@dataclass
class SkewGaussianLineProfile:
    """Asymmetric (skewed) Gaussian proxy.

    Implements a simple flux-dependent gain to mimic skewness/asymmetry effects
    in detection overlap with a symmetric filter. Positive skew slightly boosts
    low-flux signal and tapers at high flux.
    """
    sigma_A: float = 2.0
    gain: float = 1.0
    skew: float = 0.0  # dimensionless, small in magnitude (e.g., -0.5..0.5)

    def signal(self, F_true: float, lam: float) -> float:
        F = float(F_true)
        g = float(self.gain)
        s = float(self.skew)
        # Smooth boost that saturates: 1 + s * (1 - exp(-F / F0)) with F0 ~ few*sigma units in flux
        F0 = max(1e-20, 5.0 * 1.0)  # rough scale; does not depend on lam yet
        boost = 1.0 + s * (1.0 - np.exp(-F / F0))
        out = g * F * boost
        if not np.isfinite(out):
            out = 0.0
        return float(max(out, 0.0))


@dataclass
class OIIDoubletProfile:
    """Simple blended-doublet profile for [OII] 3727,3729.

    Parameters
    ----------
    sigma_A : float
        Per-component Gaussian width in Angstrom.
    gain : float
        Base gain mapping to scale with F_true.
    sep_A : float
        Component separation in Angstrom at observed wavelength scale.
    ratio : float
        Flux ratio F_3729 / F_3727 (>=0). Total F_true splits as F1 + F2.
    filter_sigma_A : float
        Width (Å) of the detection filter. Overlap efficiency increases when
        components are not fully resolved.
    """
    sigma_A: float = 2.0
    gain: float = 1.0
    sep_A: float = 2.8
    ratio: float = 1.5
    filter_sigma_A: float = 2.0

    def signal(self, F_true: float, lam: float) -> float:
        F = float(F_true)
        if not np.isfinite(F) or F <= 0:
            return 0.0
        g = float(self.gain)
        r = max(0.0, float(self.ratio))
        F1 = F * (1.0 / (1.0 + r))
        F2 = F - F1
        # Heuristic overlap efficiency between two Gaussians and a matched filter:
        # eff ≈ exp(-0.5 * (sep / sqrt(sigma^2 + filter_sigma^2))^2)
        s = max(1e-6, float(self.sigma_A))
        sf = max(1e-6, float(self.filter_sigma_A))
        sep = abs(float(self.sep_A))
        denom = np.sqrt(s * s + sf * sf)
        eff = np.exp(-0.5 * (sep / denom) ** 2)
        # Effective signal: sum of components times overlap efficiency
        out = g * (F1 + F2 * (0.5 + 0.5 * eff))  # second component partially overlaps
        if not np.isfinite(out):
            out = 0.0
        return float(max(out, 0.0))


@dataclass
class KernelEnv:
    """Noise environment passed to the kernel.

    Attributes
    ----------
    lam : float
        Observed wavelength in Angstrom.
    noise : float
        Representative flux uncertainty σ_F (e.g., from a noise bin center).
    """
    lam: float
    noise: float


def draw_signal_and_flux(
    F_true: float,
    lam: float,
    noise_env: KernelEnv,
    rng: np.random.Generator,
    *,
    profile: Optional[LineProfile] = None,
    extra_scatter: float = 0.0,
) -> Tuple[float, float, float]:
    """Return (signal, F_fit, F_error) for a latent line.

    - signal is computed via the provided `profile` (or proportional to F_true).
    - F_fit is a noisy measurement of F_true with Gaussian noise whose sigma is
      derived from the noise environment and optional extra scatter.
    - F_error is the per-object flux uncertainty used to draw F_fit.

    Notes
    -----
    The returned noise sigma respects the input `noise_env.noise` (assumed to
    already be in the same flux units as F_true) and combines it in quadrature
    with an optional `extra_scatter` term to allow simple inflations.
    """
    # Choose profile
    prof = profile if profile is not None else GaussianLineProfile()

    # Flux error σ_F from environment with optional extra scatter
    sigma_F = float(max(0.0, noise_env.noise))
    if not np.isfinite(sigma_F) or sigma_F < 0:
        sigma_F = 0.0
    if extra_scatter and np.isfinite(extra_scatter) and extra_scatter > 0:
        sigma_F = float(np.hypot(sigma_F, float(extra_scatter)))

    # Draw measured flux
    if sigma_F > 0:
        F_fit = float(F_true) + float(rng.normal(loc=0.0, scale=sigma_F))
    else:
        F_fit = float(F_true)

    # Compute a simple scalar signal via the profile
    sig = float(prof.signal(float(F_true), float(lam)))
    return sig, float(F_fit), float(sigma_F)


def build_profile_from_prior(record: Any) -> Optional[LineProfile]:
    """Construct a LineProfile from a label-scoped PriorRecord.

    Expected YAML schema under hyperparams.measurements.flux.profile:
      type: one of {'gaussian', 'skew_gaussian', 'oii_doublet'}
      params: dict of constructor kwargs
    """
    try:
        hp = getattr(record, "hyperparams", {}) or {}
        meas = hp.get("measurements", {}) or {}
        flux = meas.get("flux", {}) or {}
        prof = flux.get("profile", {}) or {}
        typ = str(prof.get("type", "")).strip().lower()
        params = dict(prof.get("params", {})) if isinstance(prof.get("params", {}), dict) else {}
        if typ == "gaussian":
            return GaussianLineProfile(**params)
        if typ == "skew_gaussian":
            return SkewGaussianLineProfile(**params)
        if typ in ("oii_doublet", "oii-doublet", "doublet"):
            return OIIDoubletProfile(**params)
    except Exception:
        return None
    return None
