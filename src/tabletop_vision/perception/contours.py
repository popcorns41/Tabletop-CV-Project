from __future__ import annotations

import cv2
import numpy as np

def find_external_contours(
        mask: np.ndarray,
) -> list[np.ndarray]:
    """Find outer contours in a binary mask."""

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    return list(contours)

def contour_area(
        contour: np.ndarray,
) -> float:
    """Return the area enclosed by a contour in pixels squared."""

    return float(
        cv2.contourArea(contour)
    )

def largest_contour(
        contours: list[np.ndarray],
) -> np.ndarray | None:
    """Return the contour with the greatest area."""

    if not contours:
        return None

    return max(
        contours,
        key=cv2.contourArea,
    )