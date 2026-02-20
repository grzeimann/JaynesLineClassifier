from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class EvidenceResult:
    """Backward-compat container for per-label measurement evidence.

    Fields
    ------
    label : str
        Label name associated with this evidence.
    log_evidence : float
        Historical name for the measurement-only log evidence. Retained for
        backward compatibility during the global rename to extra_log_likelihood.
    meta : dict
        Optional metadata for legacy callers.

    Notes
    -----
    - New code should prefer extra_log_likelihood terminology. This class
      provides a read-only alias property extra_log_likelihood to access the
      same scalar value without changing storage or wire format.
    """
    label: str
    log_evidence: float  # store log-space internally
    meta: Dict[str, Any]

    @property
    def extra_log_likelihood(self) -> float:
        """Alias for log_evidence to ease migration to new terminology."""
        return self.log_evidence


@dataclass
class SharedContext:
    # Filled by builders: cosmology tables, selection grids, caches
    cosmo: Any
    selection: Any
    caches: Dict[str, Any]
    config: Dict[str, Any]
