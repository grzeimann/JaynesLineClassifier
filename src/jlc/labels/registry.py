class LabelRegistry:
    def __init__(self, label_models):
        self._labels = {m.label: m for m in label_models}

    @property
    def labels(self):
        return list(self._labels.keys())

    def model(self, label: str):
        return self._labels[label]
