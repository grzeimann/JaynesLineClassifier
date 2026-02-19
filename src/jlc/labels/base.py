from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Sequence
import pandas as pd
from jlc.types import EvidenceResult


class LabelModel(ABC):
    label: str
    measurement_modules: Sequence

    @abstractmethod
    def log_evidence(self, row: pd.Series, ctx) -> EvidenceResult:
        """Return log P(row | label). Must marginalize nuisance params internally."""
        raise NotImplementedError

    def rate_density(self, row: pd.Series, ctx) -> float:
        """Observed-space rate density r(row) in units consistent across labels.

        Phase 1 default: return 1.0 to maintain backward compatibility.
        Implement in subclasses for physically meaningful rates.
        """
        return 1.0

    def update_hyperparams(self, df: pd.DataFrame, weights: pd.Series, ctx) -> None:
        """Optional hierarchical update (Phase 2). Default no-op."""
        return
