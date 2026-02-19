import numpy as np
from typing import Optional, Tuple, Dict, Any


class SelectionModel:
    def __init__(
        self,
        f_lim: float | None = None,
        F50: float | None = None,
        w: float | None = None,
        F50_table: Optional[Tuple[np.ndarray, np.ndarray]] | Optional[Dict[str, Any]] = None,
        w_table: Optional[Tuple[np.ndarray, np.ndarray]] | Optional[Dict[str, Any]] = None,
    ):
        """Selection model with smooth or hard completeness.

        Parameters
        ----------
        f_lim : float | None
            Legacy hard threshold. If provided and no (F50,w) are given, use C(F) = 1[F > f_lim].
        F50 : float | None
            Flux at 50% completeness for the smooth tanh model. If provided (with w), overrides f_lim.
        w : float | None
            Transition width for the smooth tanh model. Larger w makes a gentler roll-off.
        F50_table : (bins, values) tuple or dict, optional
            Optional wavelength-binned F50(λ) table. If provided, overrides scalar F50 when evaluating at a given λ.
            Bins are edges of length N+1 (Å), values length N.
        w_table : (bins, values) tuple or dict, optional
            Optional wavelength-binned w(λ) table. If provided, overrides scalar w when evaluating at a given λ.

        Behavior
        --------
        - If both F50 and w (after applying any λ-tables) are provided (finite and >0), use the smooth model:
              C(F) = 0.5 * (1 + tanh((F - F50) / w))
        - Else if f_lim is provided, use hard threshold: C(F) = 1[F > f_lim]
        - Else return C(F)=1 for all F (legacy behavior).
        """
        self.f_lim = float(f_lim) if f_lim is not None else None
        self.F50 = float(F50) if F50 is not None else None
        self.w = float(w) if w is not None else None
        self._F50_table = self._sanitize_table(F50_table)
        self._w_table = self._sanitize_table(w_table)

    # --- Table I/O helpers ---
    @staticmethod
    def save_table(path: str, bins: np.ndarray, values: np.ndarray) -> None:
        """Save a (bins, values) completeness table to disk.
        Supports .npz (preferred) or .csv with two columns: left_edge,value per row.
        """
        bins = np.asarray(bins, dtype=float)
        values = np.asarray(values, dtype=float)
        if bins.ndim != 1 or values.ndim != 1 or bins.size != values.size + 1:
            raise ValueError("Invalid table: bins must have length N+1 and values length N")
        if path.lower().endswith(".npz"):
            import numpy as _np
            _np.savez_compressed(path, bins=bins, values=values)
        elif path.lower().endswith(".csv"):
            import csv
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["bin_left", "value"])  # header
                for i in range(values.size):
                    w.writerow([bins[i], values[i]])
                # optionally include rightmost edge as a comment-like last row (NaN,value)
        else:
            raise ValueError("Unsupported file extension for table; use .npz or .csv")

    @staticmethod
    def load_table(path: str) -> Dict[str, np.ndarray]:
        """Load a completeness table saved by save_table() or a compatible producer.
        Returns a dict with keys {bins, values}.
        """
        if path.lower().endswith(".npz"):
            import numpy as _np
            with _np.load(path, allow_pickle=False) as data:
                bins = _np.asarray(data["bins"], dtype=float)
                values = _np.asarray(data["values"], dtype=float)
        elif path.lower().endswith(".csv"):
            import numpy as _np
            import csv
            lefts = []
            vals = []
            with open(path, "r", newline="") as f:
                r = csv.reader(f)
                header = next(r, None)
                for row in r:
                    if not row:
                        continue
                    try:
                        lefts.append(float(row[0]))
                        vals.append(float(row[1]))
                    except Exception:
                        continue
            lefts = _np.asarray(lefts, dtype=float)
            vals = _np.asarray(vals, dtype=float)
            if lefts.size == 0 or vals.size == 0 or lefts.size != vals.size:
                raise ValueError("CSV table must have at least one row and matching columns")
            # reconstruct rightmost edge by extending the last bin by the median width
            widths = _np.diff(lefts)
            wmed = float(_np.median(widths)) if widths.size > 0 else 1.0
            bins = _np.concatenate([lefts, [lefts[-1] + wmed]])
            values = vals
        else:
            raise ValueError("Unsupported table file; use .npz or .csv")
        if bins.ndim != 1 or values.ndim != 1 or bins.size != values.size + 1:
            raise ValueError("Loaded table has invalid shapes (bins N+1, values N)")
        return {"bins": bins, "values": values}

    @staticmethod
    def _sanitize_table(tbl: Optional[Tuple[np.ndarray, np.ndarray]] | Optional[Dict[str, Any]]):
        if tbl is None:
            return None
        try:
            if isinstance(tbl, tuple) and len(tbl) == 2:
                bins = np.asarray(tbl[0], dtype=float)
                vals = np.asarray(tbl[1], dtype=float)
            elif isinstance(tbl, dict):
                bins = np.asarray(tbl.get("bins"), dtype=float)
                vals = np.asarray(tbl.get("values"), dtype=float)
            else:
                return None
            if bins.ndim != 1 or vals.ndim != 1:
                return None
            if bins.size != vals.size + 1 or bins.size < 2:
                return None
            if not (np.all(np.isfinite(bins)) and np.all(np.isfinite(vals))):
                return None
            if not np.all(np.diff(bins) > 0):
                return None
            return {"bins": bins, "values": vals}
        except Exception:
            return None

    @staticmethod
    def _value_from_table(lam: float, table: Optional[Dict[str, np.ndarray]]):
        if table is None or not np.isfinite(lam):
            return None
        try:
            bins = table["bins"]
            vals = table["values"]
            if lam < bins[0] or lam > bins[-1]:
                return None
            idx = int(np.searchsorted(bins, lam, side="right") - 1)
            idx = int(np.clip(idx, 0, vals.size - 1))
            v = float(vals[idx])
            return v if np.isfinite(v) else None
        except Exception:
            return None

    def completeness(self, F: np.ndarray, wave_obs: float) -> np.ndarray:
        """Return selection completeness in [0,1] for each flux value at given wave.

        Notes
        -----
        - Vectorized over F.
        - Supports optional wavelength dependence via binned F50(λ) and w(λ) tables.
        - Guarantees outputs in [0,1].
        """
        F = np.asarray(F, dtype=float)
        # Effective per-wavelength parameters if tables provided
        F50_eff = self._value_from_table(wave_obs, self._F50_table)
        if F50_eff is None:
            F50_eff = self.F50
        w_eff = self._value_from_table(wave_obs, self._w_table)
        if w_eff is None:
            w_eff = self.w
        # Smooth tanh model if configured and sane
        if F50_eff is not None and w_eff is not None and np.isfinite(F50_eff) and np.isfinite(w_eff) and w_eff > 0:
            x = (F - F50_eff) / w_eff
            C = 0.5 * (1.0 + np.tanh(x))
            # numerical safety: clamp to [0,1]
            return np.clip(C, 0.0, 1.0)
        # Hard threshold fallback
        if self.f_lim is not None and np.isfinite(self.f_lim):
            return (F > self.f_lim).astype(float)
        # Legacy: always selected
        return np.ones_like(F)
