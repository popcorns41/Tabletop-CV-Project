import cv2 
import numpy as np
import pytest

from tabletop_vision.perception.colour import (
    HSVRange,
    create_hsv_mask,
)

def test_hsv_range_rejects_invalid_hue() -> None:
    with pytest.raises(ValueError):
        HSVRange(
            hue_min=-1,
            saturation_min=0,
            value_min=0,
            hue_max=100,
            saturation_max=255,
            value_max=255,
        )


def test_mask_accepts_matching_pixel() -> None:
    hsv_pixel = np.array(
        [[[100, 200, 200]]],
        dtype=np.uint8,
    )

    bgr_pixel = cv2.cvtColor(
        hsv_pixel,
        cv2.COLOR_HSV2BGR,
    )

    colour_range = HSVRange(
        hue_min=90,
        saturation_min=150,
        value_min=150,
        hue_max=110,
        saturation_max=255,
        value_max=255,
    )

    mask = create_hsv_mask(
        bgr_pixel,
        colour_range,
    )

    assert mask[0, 0] == 255