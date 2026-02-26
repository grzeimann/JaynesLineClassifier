from __future__ import annotations
import os
import sys
import math
from typing import Any, Optional

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None  # type: ignore

try:
    import resource  # type: ignore
except Exception:  # pragma: no cover
    resource = None  # type: ignore


def format_bytes(nbytes: float) -> str:
    try:
        n = float(nbytes)
    except Exception:
        return str(nbytes)
    if not math.isfinite(n) or n < 0:
        return str(nbytes)
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    i = 0
    while n >= 1024.0 and i < len(units) - 1:
        n /= 1024.0
        i += 1
    return f"{n:.2f} {units[i]}"


def array_nbytes(arr: Any) -> int:
    try:
        import numpy as np  # local
        if isinstance(arr, np.ndarray):
            return int(arr.nbytes)
    except Exception:
        pass
    try:
        # Fallback for python containers: rough estimate
        return sys.getsizeof(arr)
    except Exception:
        return 0


def mem_info() -> dict:
    """Return a dict with RSS and peak RSS in bytes when available."""
    info = {"rss": None, "rss_peak": None}
    # psutil preferred for current RSS
    try:
        if psutil is not None:
            p = psutil.Process(os.getpid())
            info["rss"] = int(p.memory_info().rss)
    except Exception:
        pass
    # resource for peak on POSIX
    try:
        if resource is not None:
            r = resource.getrusage(resource.RUSAGE_SELF)
            peak_kb = getattr(r, "ru_maxrss", 0) or 0
            # ru_maxrss is KB on Linux, bytes on macOS; normalize via heuristic
            # Assume KB unless absurdly small
            if peak_kb < 1e6:  # likely KB
                info["rss_peak"] = int(peak_kb * 1024)
            else:
                info["rss_peak"] = int(peak_kb)
    except Exception:
        pass
    return info


def log_mem(tag: str, extras: Optional[dict] = None) -> None:
    """Log a one-line memory status message via project logger."""
    try:
        from jlc.utils.logging import log
    except Exception:  # pragma: no cover
        def log(x: str):  # type: ignore
            print(x)
    info = mem_info()
    parts = [f"rss={format_bytes(info.get('rss') or 0)}"]
    if info.get("rss_peak") is not None:
        parts.append(f"peak={format_bytes(info['rss_peak'])}")
    if extras:
        for k, v in extras.items():
            if isinstance(v, (int, float)):
                parts.append(f"{k}={format_bytes(v)}")
            else:
                parts.append(f"{k}={v}")
    log(f"[jlc.mem] {tag}: " + ", ".join(parts))


# ==========================
# Completeness diagnostics
# ==========================

class CompletenessTracer:
    """Lightweight tracer to observe completeness values and exceptions.

    When enabled, callers can record summary stats for C(F, λ) per label and
    track exception counts without altering core logic.
    """
    def __init__(self) -> None:
        # key: (label), value: dict with aggregates
        self.enabled: bool = False
        self._counts: dict[str, int] = {}
        self._nz_approx1: dict[str, int] = {}  # number of arrays with ~all ones
        self._nz_approx0: dict[str, int] = {}  # number of arrays with ~all zeros
        self._elem: dict[str, int] = {}        # total elements observed
        self._exc_provider: dict[str, int] = {}  # exceptions inside providers per label
        self._exc_integrand: dict[str, int] = {}  # exceptions at integrand callsite per label

    def reset(self) -> None:
        self._counts.clear(); self._nz_approx1.clear(); self._nz_approx0.clear()
        self._elem.clear(); self._exc_provider.clear(); self._exc_integrand.clear()

    def observe(self, F, C, label: str, lam: float, noise: float | None = None) -> None:
        if not self.enabled:
            return
        try:
            import numpy as np
            L = str(label)
            C_arr = np.asarray(C, dtype=float)
            n = int(C_arr.size)
            if n <= 0:
                return
            self._counts[L] = self._counts.get(L, 0) + 1
            self._elem[L] = self._elem.get(L, 0) + n
            # Degeneracy checks
            finite = np.isfinite(C_arr)
            if np.any(finite):
                c = C_arr[finite]
                frac1 = float(np.mean(np.abs(c - 1.0) < 1e-6))
                frac0 = float(np.mean(np.abs(c - 0.0) < 1e-6))
                if frac1 > 0.95:  # nearly all ones
                    self._nz_approx1[L] = self._nz_approx1.get(L, 0) + 1
                if frac0 > 0.95:  # nearly all zeros
                    self._nz_approx0[L] = self._nz_approx0.get(L, 0) + 1
        except Exception:
            pass

    def on_exception(self, where: str, label: str) -> None:
        if not self.enabled:
            return
        L = str(label)
        if where == "provider":
            self._exc_provider[L] = self._exc_provider.get(L, 0) + 1
        else:
            self._exc_integrand[L] = self._exc_integrand.get(L, 0) + 1

    def summarize(self) -> None:
        if not self.enabled:
            return
        try:
            from jlc.utils.logging import log as _log
        except Exception:
            def _log(x: str):  # type: ignore
                print(x)
        labels = sorted(set(list(self._counts.keys()) + list(self._exc_provider.keys()) + list(self._exc_integrand.keys())))
        if len(labels) == 0:
            _log("[jlc.simulate] CompletenessTracer: no observations recorded")
            return
        parts = []
        for L in labels:
            obs = self._counts.get(L, 0)
            n = self._elem.get(L, 0)
            a1 = self._nz_approx1.get(L, 0)
            a0 = self._nz_approx0.get(L, 0)
            eprov = self._exc_provider.get(L, 0)
            eint = self._exc_integrand.get(L, 0)
            parts.append(f"{L}: obs={obs}, elems={n}, ~all1={a1}, ~all0={a0}, exc(provider)={eprov}, exc(integrand)={eint}")
        _log("[jlc.simulate] Completeness summary → " + " | ".join(parts))


# Module-level singleton and helpers
_current_tracer: CompletenessTracer = CompletenessTracer()

def get_completeness_tracer() -> CompletenessTracer:
    return _current_tracer


def enable_completeness_tracing(enabled: bool = True, reset: bool = True) -> None:
    _current_tracer.enabled = bool(enabled)
    if enabled and reset:
        _current_tracer.reset()
