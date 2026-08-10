from __future__ import annotations

from dataclasses import dataclass

import numpy as np

@dataclass(frozen=True, slots=True)
class SegmentationPrediction:
    """One predicted object instance from a learned model."""

    class_id: int
    class_name: str
    confidence: float
    mask: np.ndarray
    polygon: np.ndarray