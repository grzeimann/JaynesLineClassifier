from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path


@dataclass
class PriorRecord:
    """Persistent snapshot of hyperparameters and provenance.

    Scope can be one of:
      - 'label': per-label hyperparameters, including population and per-measurement priors
      - 'global': engine- or run-level hyperparameters
      - 'selection': selection-model specific hyperparameters
    """
    name: str
    scope: str
    label: Optional[str] = None
    hyperparams: Dict[str, Any] = field(default_factory=dict)
    source: str = "manual"
    notes: str = ""
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


# --- Simple YAML helpers ---
try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional until user installs pyyaml
    yaml = None


def save_prior_record(record: PriorRecord, path: str | Path) -> None:
    """Save a PriorRecord to YAML.

    If PyYAML is unavailable, falls back to writing a minimal repr via stdlib.
    """
    p = Path(path)
    data = {
        "name": record.name,
        "scope": record.scope,
        "label": record.label,
        "hyperparams": record.hyperparams,
        "source": record.source,
        "notes": record.notes,
        "created_at": record.created_at,
    }
    if yaml is not None:
        p.write_text(yaml.safe_dump(data, sort_keys=False))
    else:
        # Very simple YAML-like serialization to preserve readability
        import json
        p.write_text(json.dumps(data, indent=2))


def load_prior_record(path: str | Path) -> PriorRecord:
    """Load a PriorRecord from YAML (or JSON fallback)."""
    p = Path(path)
    txt = p.read_text()
    if yaml is not None:
        data = yaml.safe_load(txt)
    else:
        import json
        data = json.loads(txt)
    return PriorRecord(**data)


def apply_prior_to_label(record: PriorRecord, label_model) -> None:
    """Apply a label-scoped PriorRecord to a LabelModel.

    This performs a shallow merge of hyperparameters. Projects can extend this
    function to perform deep merges as needed.
    """
    assert record.scope == "label", "PriorRecord scope must be 'label' to apply to a label model"
    assert getattr(label_model, "label", None) == record.label, "PriorRecord label mismatch"
    try:
        hp_cur = label_model.get_hyperparams_dict()
    except Exception:
        hp_cur = {}
    # Shallow merge: record.hyperparams entries override existing keys
    merged = dict(hp_cur)
    merged.update(record.hyperparams or {})
    try:
        label_model.set_hyperparams(**merged)
    except Exception:
        # As a fallback, store on the model for visibility
        try:
            label_model.hyperparams = merged
        except Exception:
            pass
