import numpy as np
import pandas as pd


class JaynesianEngine:
    def __init__(self, registry, ctx):
        self.registry = registry
        self.ctx = ctx

    def compute_log_evidence_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for label in self.registry.labels:
            model = self.registry.model(label)
            out[f"logZ_{label}"] = [
                model.log_evidence(row, self.ctx).log_evidence
                for _, row in df.iterrows()
            ]
        return out

    def normalize_posteriors(self, df: pd.DataFrame, log_prior_weights: dict | None = None) -> pd.DataFrame:
        out = df.copy()
        labels = self.registry.labels

        logW = (
            np.array([log_prior_weights.get(L, 0.0) for L in labels], dtype=float)
            if log_prior_weights
            else np.zeros(len(labels))
        )
        logZ = np.vstack([out[f"logZ_{L}"].values for L in labels]).T  # (N, K)
        logP_unnorm = logZ + logW[None, :]

        # log-softmax
        m = np.max(logP_unnorm, axis=1, keepdims=True)
        P = np.exp(logP_unnorm - m)
        denom = np.sum(P, axis=1, keepdims=True)
        # Avoid division by zero if all -inf
        denom = np.where(denom <= 0, 1.0, denom)
        P = P / denom

        for j, L in enumerate(labels):
            out[f"p_{L}"] = P[:, j]
        return out
