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

def filter_contours_by_area(
        contours: list[np.ndarray],
        minimum_area: float,
        maximum_area: float | None = None,
) -> list[np.ndarray]:
    """Return contours whose areas lie within the requested range."""

    if minimum_area < 0:
        raise ValueError(
            "minimum_area must not be negative."
        )

    if (
        maximum_area is not None
        and maximum_area < minimum_area
    ):
        raise ValueError(
            "maximum_area must not be smaller than minimum_area."
        )

    accepted: list[np.ndarray] = []

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < minimum_area:
            continue

        if (
            maximum_area is not None
            and area > maximum_area
        ):
            continue

        accepted.append(contour)

    return accepted

def contour_centroid(
        contour: np.ndarray,
) -> tuple[int, int] | None:
    """Return the centroid of a contour in pixel coordinates."""

    moments = cv2.moments(contour)
    area = moments["m00"]

    if area == 0:
        return None

    centre_x = int(
        round(moments["m10"] / area)
    )

    centre_y = int(
            round(moments["m01"] / area)
        )

    return centre_x, centre_y

