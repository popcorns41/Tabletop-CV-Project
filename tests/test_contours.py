import cv2
import numpy as np

from tabletop_vision.perception import (
    filter_contours_by_area,
    find_external_contours,
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