from __future__ import annotations

from dataclasses import dataclass

import numpy as np

@dataclass(frozen=True, slots=True)
class RotatedRectangle:
    """Rotated bounding rectangle around a detected object."""

    centre: tuple[float, float]
    width: float
    height: float
    angle_degrees: float
    corners: np.ndarray