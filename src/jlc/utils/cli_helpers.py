from __future__ import annotations
"""
Small CLI helper loaders used by multiple commands.

This module centralizes tiny utilities previously defined inline in the CLI to
reduce duplication and improve readability. No behavioral changes.
"""
from typing import Any, Dict, Tuple
from jlc.utils.logging import log


def load_ra_dec_factor(spec: str | None):
    """Load a RA/Dec modulation function from a 'module:function' spec.

    The function should have signature g(ra, dec, lam) -> float in [0,1].
    Returns None if spec is None or loading fails.
    """
    if not spec:
        return None
    try:
        mod_name, func_name = (spec.split(":", 1) + [None])[:2]
        if not mod_name or not func_name:
            return None
        import importlib
        mod = importlib.import_module(mod_name)
        fn = getattr(mod, func_name, None)
        if callable(fn):
            return fn
    except Exception as e:
        try:
            log(f"[jlc] Warning: failed to load ra_dec_factor '{spec}': {e}")
        except Exception:
            pass
    return None


def load_completeness_tables(args: Any, caller: str = "jlc") -> Tuple[Dict[str, Any] | None, Dict[str, Any] | None]:
    """Shared loader for F50_table and w_table CLI args.

    Returns a tuple (F50_table, w_table), where each is either a dict with keys
    {bins, values} or None if not provided or if loading failed. Logs warnings
    with a standard prefix that includes the caller context (e.g., jlc.classify).
    """
    F50_table = None
    w_table = None
    prefix = f"[{caller}]"
    if getattr(args, "F50_table", None):
        try:
            from jlc.selection.base import SelectionModel as _Sel
            F50_table = _Sel.load_table(args.F50_table)
        except Exception as e:
            try:
                log(f"{prefix} Warning: failed to load F50_table from {args.F50_table}: {e}")
            except Exception:
                pass
            F50_table = None
    if getattr(args, "w_table", None):
        try:
            from jlc.selection.base import SelectionModel as _Sel
            w_table = _Sel.load_table(args.w_table)
        except Exception as e:
            try:
                log(f"{prefix} Warning: failed to load w_table from {args.w_table}: {e}")
            except Exception:
                pass
            w_table = None
    return F50_table, w_table
