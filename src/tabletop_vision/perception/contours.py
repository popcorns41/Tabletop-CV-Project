from __future__ import annotations

import cv2
import numpy as np

import math

from tabletop_vision.perception.models import (
    RotatedRectangle,
)

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

def contour_rotated_rectangle(
        contour: np.ndarray,
) -> RotatedRectangle:
    """Return the minimum-area rectangle and long-axis orientation."""

    rectangle = cv2.minAreaRect(contour)

    centre, _, _ = rectangle

    corners = cv2.boxPoints(
        rectangle
    ).astype(np.float32)

    longest_length = -1.0
    shortest_length = math.inf

    longest_dx = 0.0
    longest_dy = 0.0

    for index in range(4):
        start = corners[index]
        end = corners[(index + 1) % 4]

        dx = float(end[0] - start[0])
        dy = float(end[1] - start[1])

        length = math.hypot(
            dx,
            dy,
        )

        if length > longest_length:
            longest_length = length
            longest_dx = dx
            longest_dy = dy

        if length < shortest_length:
            shortest_length = length

    angle_degrees = math.degrees(
        math.atan2(
            longest_dy,
            longest_dx,
        )
    )

    #An axis has 180-degree symmetry.
    #Normalise it into [-90, 90].

    while angle_degrees >= 90.0:
        angle_degrees -= 180.0

    while angle_degrees < -90.0:
        angle_degrees += 180.0

    return RotatedRectangle(

        centre= (
            float(centre[0]),
            float(centre[1])
        ),
        width = longest_length,
        height= shortest_length,
        angle_degrees= angle_degrees,
        corners= corners,
    )

def long_axis_endpoints(
        rotated_rectangle: RotatedRectangle,
) -> tuple[tuple[int,int],tuple[int,int]]:
    """Return the long axis direction end points of a given Rotated Rectangle"""

    angle_radians = math.radians(
        rotated_rectangle.angle_degrees
    )

    centre_x = int(
        round(rotated_rectangle.centre[0])
    )

    centre_y = int(
        round(rotated_rectangle.centre[1])
    )

    half_length = (
        rotated_rectangle.width / 2.0
    )

    dx = math.cos(angle_radians) * half_length
    dy = math.sin(angle_radians) * half_length

    start_point = (
        int(round(centre_x - dx)),
        int(round(centre_y -dy)),
    )

    end_point = (
        int(round(centre_x + dx)),
        int(round(centre_y + dy)),
    )

    return start_point, end_point

def segmentation_polygon_to_contour(
        polygon: np.ndarray,
) -> np.ndarray:
    """Convert a segmentation polygon into an OpenCV contour."""

    if polygon.ndim != 2:
        raise ValueError(
            "Polygon must have shape (N, 2)."
        )

    if polygon.shape[1] != 2:
        raise ValueError(
            "Polygon points must contain x and y."
        )

    if len(polygon) < 3:
        raise ValueError(
            "Polygon must contain at least three points."
        )

    return polygon.reshape(
        -1,
        1,
        2,
    ).astype(np.float32)