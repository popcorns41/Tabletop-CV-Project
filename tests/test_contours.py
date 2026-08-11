import cv2
import numpy as np
import pytest

from tabletop_vision.perception import (
    filter_contours_by_area,
    find_external_contours,
    contour_centroid,
    contour_rotated_rectangle,
    polygon_to_contour
)

def test_find_external_contours_find_blob() -> None:
    mask = np.zeros(
        (100,100),
        dtype=np.uint8,
    )

    cv2.rectangle(
        mask,
        (20, 20),
        (60, 60),
        255,
        thickness=-1,
    )

    contours = find_external_contours(mask)

    assert len(contours) == 1

def test_filter_contours_remove_small_blob() -> None:
    mask = np.zeros(
        (100, 100),
        dtype=np.uint8,
    )

    #Tiny noise.
    cv2.rectangle(
        mask,
        (30, 30),
        (70, 70),
        255,
        thickness=-1,
    )

    contours = find_external_contours(mask)

    filtered = filter_contours_by_area(
        contours,
        minimum_area=100.0,
    )

    assert len(contours) == 1
    assert len(filtered) == 1


def test_contour_centroid_finds_rectangle_centre() -> None:
    mask = np.zeros(
        (100, 100),
        dtype=np.uint8,
    )

    cv2.rectangle(
        mask,
        (20, 30),
        (60, 70),
        255,
        thickness=-1,
    )

    contours = find_external_contours(mask)

    centroid = contour_centroid(
        contours[0]
    )

    assert centroid == (40, 50)

def test_rotated_rectangle_detects_horizontal_axis() -> None:
    contour = np.array(
        [
            [[20, 40]],
            [[80, 40]],
            [[80, 60]],
            [[20, 60]],
        ],
        dtype=np.int32,
    )

    rectangle = contour_rotated_rectangle(
        contour
    )

    assert abs(
        rectangle.angle_degrees
    ) < 1.0

    assert rectangle.width == pytest.approx(
        60.0,
        abs=1.0,
    )

    assert rectangle.height == pytest.approx(
        20.0,
        abs=1.0,
    )

def test_rotated_rectangle_detects_diagonal_axis() -> None:
    source_rectangle = (
        (50.0, 50.0),
        (60.0, 20.0),
        30.0,
    )

    corners = cv2.boxPoints(
        source_rectangle
    )

    contour = corners.reshape(
        -1,
        1,
        2,
    ).astype(np.int32)

    rectangle = contour_rotated_rectangle(
        contour
    )

    assert rectangle.angle_degrees == pytest.approx(
        30.0,
        abs=2.0,
    )

def test_segmentation_polygon_to_contour() -> None:
    polygon = np.array(
        [
            [20.0, 30.0],
            [80.0, 30.0],
            [80.0, 70.0],
            [20.0, 70.0],
        ],
        dtype=np.float32
    )

    contour = polygon_to_contour(
        polygon
    )

    assert contour.shape == (
        4,
        1,
        2,
    )

    centroid = contour_centroid(
        contour
    )

    assert centroid == (
        50,
        50,
    )

def test_segmentation_polygon_rejects_too_few_points() -> None:
    polygon = np.array(
        [
            [10.0, 10.0],
            [20.0, 20.0],
        ],
        dtype=np.float32,
    )

    with pytest.raises(ValueError):
        polygon_to_contour(
            polygon
        )