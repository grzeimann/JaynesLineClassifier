from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class EvidenceResult:
    label: str
    log_evidence: float  # store log-space internally
    meta: Dict[str, Any]


@dataclass
class SharedContext:
    # Filled by builders: cosmology tables, selection grids, caches
    cosmo: Any
    selection: Any
    caches: Dict[str, Any]
    config: Dict[str, Any]
