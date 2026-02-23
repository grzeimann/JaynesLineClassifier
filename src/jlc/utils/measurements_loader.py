from __future__ import annotations
from typing import Any, Dict, List, Sequence


def _import_string(path: str):
    """Lightweight import_string("pkg.mod:Class") utility.

    Accepts either "pkg.mod:Class" or "pkg.mod.Class" forms.
    """
    import importlib
    if ":" in path:
        mod_name, attr = path.split(":", 1)
    elif path.count(".") >= 1:
        # split on last dot
        mod_name, attr = path.rsplit(".", 1)
    else:
        raise ImportError(f"Invalid import path '{path}'. Expected 'pkg.mod:Class' or 'pkg.mod.Class'.")
    mod = importlib.import_module(mod_name)
    try:
        return getattr(mod, attr)
    except AttributeError as e:
        raise ImportError(f"Module '{mod_name}' has no attribute '{attr}'") from e


def load_measurements_from_config(cfg: Dict[str, Any] | None, prior_record: Any | None = None) -> Sequence[Any]:
    """Load measurement module instances from a head configuration dict.

    Schema example:
      cfg = {
        "measurements": {
          "flux": {
            "module": "jlc.measurements.flux:FluxMeasurement",
            "catalog_columns": {"value": "flux_hat", "error": "flux_err"},
            "latent_key": "F_true",
          },
          "wavelength": {
            "module": "jlc.measurements.wavelength:WavelengthMeasurement",
            "catalog_columns": {"value": "wave_obs", "error": "wave_err"},
            "latent_key": "wave_true",
          },
        }
      }

    If a PriorRecord is provided, attempts to apply per-measurement blocks under
    record.hyperparams["measurements"][<name>]["noise"/"prior"]["params"].
    """
    if not cfg or "measurements" not in cfg or not isinstance(cfg["measurements"], dict):
        return []
    meas_cfg = cfg["measurements"]
    hp_meas = {}
    if prior_record is not None:
        try:
            hp = getattr(prior_record, "hyperparams", {}) or {}
            hp_meas = dict(hp.get("measurements", {})) if isinstance(hp.get("measurements", {}), dict) else {}
        except Exception:
            hp_meas = {}
    out: List[Any] = []
    for name, mcfg in meas_cfg.items():
        try:
            cls_path = mcfg.get("module")
            if not cls_path:
                continue
            cls = _import_string(str(cls_path))
            # Build instance with possible noise/prior hyperparams
            noise_hp = {}
            prior_hp = {}
            blk = hp_meas.get(name, {}) if isinstance(hp_meas, dict) else {}
            try:
                nz = blk.get("noise", {}) or {}
                noise_hp = dict(nz.get("params", {})) if isinstance(nz.get("params", {}), dict) else dict(nz)
            except Exception:
                noise_hp = {}
            try:
                pr = blk.get("prior", {}) or {}
                prior_hp = dict(pr.get("params", {})) if isinstance(pr.get("params", {}), dict) else dict(pr)
            except Exception:
                prior_hp = {}
            inst = cls(noise_hyperparams=noise_hp, prior_hyperparams=prior_hp)
            # Optional metadata overrides
            cols = mcfg.get("catalog_columns")
            if isinstance(cols, dict):
                try:
                    # Standard two-field shorthand (value, error); else flatten values
                    if "value" in cols and "error" in cols:
                        inst.catalog_columns = (str(cols["value"]), str(cols["error"]))
                    else:
                        inst.catalog_columns = tuple(str(v) for v in cols.values())
                except Exception:
                    pass
            if "latent_key" in mcfg:
                try:
                    inst.latent_key = str(mcfg["latent_key"]) if mcfg["latent_key"] else inst.latent_key
                except Exception:
                    pass
            out.append(inst)
        except Exception:
            # Skip misconfigured entries to remain robust
            continue
    return out
