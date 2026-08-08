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

@dataclass(frozen=True, slots=True)
class ObjectDetection:
    """Geometric description of a detected object in image space."""

    contour: np.ndarray
    centroid: tuple[int, int]
    area_pixels_squared: float
    rotated_rectangle: RotatedRectangle