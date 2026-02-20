class LabelRegistry:
    def __init__(self, label_models):
        self._labels = {m.label: m for m in label_models}

    @property
    def labels(self):
        return list(self._labels.keys())

    def model(self, label: str):
        return self._labels[label]

    # Convenience properties for common LFs
    @property
    def lae_lf(self):
        try:
            m = self._labels.get("lae")
            return getattr(m, "lf", None)
        except Exception:
            return None

    @property
    def oii_lf(self):
        try:
            m = self._labels.get("oii")
            return getattr(m, "lf", None)
        except Exception:
            return None
