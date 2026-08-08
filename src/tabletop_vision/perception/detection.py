from __future__ import annotations

import numpy as np

from tabletop_vision.perception.contours import (
    contour_area,
    contour_centroid,
    contour_rotated_rectangle,
    filter_contours_by_area,
    find_external_contours,
    largest_contour,
)

from tabletop_vision.perception.models import (
    ObjectDetection,
)

def detect_largest_object(
        mask: np.ndarray,
        minimum_area: float,
        maximum_area: float | None = None,
) -> ObjectDetection | None:
    """Detect the largest valid foreground object in a binary mask."""

    contours = find_external_contours(
        mask
    )

    valid_contours = filter_contours_by_area(
        contours,
        minimum_area=minimum_area,
        maximum_area=maximum_area,
    )

    target = largest_contour(
        valid_contours
    )

    if target is None:
        return None

    centroid = contour_centroid(
        target
    )

    if centroid is None:
        return None

    rotated_rectangle = contour_rotated_rectangle(
        target
    )

    return ObjectDetection(
        contour=target,
        centroid=centroid,
        area_pixels_squared=contour_area(target),
        rotated_rectangle=rotated_rectangle,
    )
