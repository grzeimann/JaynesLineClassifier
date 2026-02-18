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

    def update_hyperparams(self, df: pd.DataFrame, weights: pd.Series, ctx) -> None:
        """Optional hierarchical update (Phase 2). Default no-op."""
        return
