from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any
import pandas as pd


class MeasurementModule(ABC):
    name: str

    @abstractmethod
    def log_likelihood(self, row: pd.Series, latent: Dict[str, Any], ctx) -> float:
        """Return log P(measurement | latent, label)."""
        raise NotImplementedError
