"""Shared hazard-score definitions for the two environmental case studies.

The canonical full score puts every endpoint on the same Development-OOF
empirical-percentile scale. Higher percentiles always mean greater concern.
The toxicity component is the maximum of the three inverse concentration
percentiles, and P, B and T are combined by geometric mean.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd


P_TASK = "P_screen_not_ready_binary"
B_TASK = "B_LOGBCF_REGRESSION"
EC50_TASK = "T_EC50_MG_L_REGRESSION"
LC50_TASK = "T_LC50_MG_L_REGRESSION"
NOEC_TASK = "T_NOEC_MG_L_REGRESSION"
TOXICITY_TASKS = (EC50_TASK, LC50_TASK, NOEC_TASK)
TASK_ORDER = (P_TASK, B_TASK, *TOXICITY_TASKS)

CANONICAL_HAZARD_FORMULA = "development_oof_percentiles__toxicity_max__pbt_geometric_mean"


def empirical_hazard_percentile(
    values: np.ndarray,
    reference: np.ndarray,
    *,
    high_is_hazard: bool,
) -> np.ndarray:
    """Map predictions to smoothed empirical hazard percentiles."""
    values = np.asarray(values, dtype=float)
    reference = np.sort(np.asarray(reference, dtype=float))
    if not len(reference) or not np.isfinite(reference).all():
        raise ValueError("Development reference must be non-empty and finite")
    if not np.isfinite(values).all():
        raise ValueError("Predictions must be finite")
    if high_is_hazard:
        ranks = np.searchsorted(reference, values, side="right")
    else:
        ranks = len(reference) - np.searchsorted(reference, values, side="left")
    return (ranks + 0.5) / (len(reference) + 1.0)


def load_development_references(path: str | Path) -> dict[str, np.ndarray]:
    """Load the selected-model Development OOF prediction distribution."""
    frame = pd.read_csv(path, low_memory=False)
    required = {"task", "y_score_calibrated"}
    if not required.issubset(frame.columns):
        raise RuntimeError(f"Development OOF schema mismatch: {required - set(frame.columns)}")
    references: dict[str, np.ndarray] = {}
    for task in TASK_ORDER:
        values = pd.to_numeric(
            frame.loc[frame["task"].eq(task), "y_score_calibrated"], errors="coerce"
        ).dropna().to_numpy(float)
        if not len(values):
            raise RuntimeError(f"No Development OOF reference values for {task}")
        references[task] = values
    return references


def harmonized_full_hazard(
    raw_predictions: Mapping[str, np.ndarray],
    references: Mapping[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Return endpoint percentiles, max-toxicity component and P/B/T score."""
    missing_predictions = set(TASK_ORDER) - set(raw_predictions)
    missing_references = set(TASK_ORDER) - set(references)
    if missing_predictions or missing_references:
        raise KeyError(
            f"Missing hazard inputs: predictions={sorted(missing_predictions)}, "
            f"references={sorted(missing_references)}"
        )
    endpoint_hazard = {
        task: empirical_hazard_percentile(
            raw_predictions[task],
            references[task],
            high_is_hazard=task not in TOXICITY_TASKS,
        )
        for task in TASK_ORDER
    }
    toxicity = np.max(
        np.column_stack([endpoint_hazard[task] for task in TOXICITY_TASKS]), axis=1
    )
    hazard = np.cbrt(endpoint_hazard[P_TASK] * endpoint_hazard[B_TASK] * toxicity)
    if not (np.isfinite(hazard).all() and ((hazard > 0) & (hazard < 1)).all()):
        raise RuntimeError("Canonical hazard score must be finite and strictly between zero and one")
    return endpoint_hazard, toxicity, hazard
