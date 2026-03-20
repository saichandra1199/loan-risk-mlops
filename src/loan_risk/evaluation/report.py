"""Evaluation report: structured summary of a model evaluation run."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class EvaluationReport:
    """Complete evaluation report for a single training run.

    Serialisable to JSON for storage in reports/evaluation/.
    """

    run_id: str
    model_name: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    # Core metrics
    val_metrics: dict[str, float] = field(default_factory=dict)
    test_metrics: dict[str, float] = field(default_factory=dict)

    # Threshold
    threshold: float = 0.5

    # Hyperparameters used
    params: dict[str, Any] = field(default_factory=dict)

    # SHAP feature importance (top N)
    top_features: list[dict[str, Any]] = field(default_factory=list)

    # Bias audit results
    slice_metrics: dict[str, list[dict]] = field(default_factory=dict)
    disparate_impact: dict[str, list[dict]] = field(default_factory=dict)

    # Promotion decision
    promoted: bool = False
    promotion_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def save(self, output_dir: str = "reports/evaluation") -> Path:
        """Save report to a timestamped JSON file."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        filename = f"report_{self.run_id[:8]}_{self.timestamp[:10]}.json"
        path = output_path / filename
        path.write_text(self.to_json())
        return path

    @classmethod
    def load(cls, path: str | Path) -> "EvaluationReport":
        """Load a report from a JSON file."""
        data = json.loads(Path(path).read_text())
        return cls(**data)

    def passes_promotion_gate(self, auc_threshold: float) -> bool:
        """Return True if test AUC meets the promotion threshold."""
        test_auc = self.test_metrics.get("auc_roc", 0.0)
        passes = test_auc >= auc_threshold
        self.promoted = passes
        self.promotion_reason = (
            f"Test AUC {test_auc:.4f} {'≥' if passes else '<'} threshold {auc_threshold:.4f}"
        )
        return passes
