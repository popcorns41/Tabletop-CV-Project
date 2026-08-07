from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class HSVRange:
    """Inclusive lower and upper bounds for HSV colour segmentation."""

    hue_min: int
    saturation_min: int
    value_min: int

    hue_max: int
    saturation_max: int
    value_max: int

    def __post_init__(self) -> None:
        if not 0 <= self.hue_min <= 179:
            raise ValueError("hue_min must be between 0 and 179.")

        if not 0 <= self.hue_max <= 179:
            raise ValueError("hue_max must be between 0 and 179.")

        if not 0 <= self.saturation_min <= 255:
            raise ValueError(
                "saturation_min must be between 0 and 255."
            )

        if not 0 <= self.saturation_max <= 255:
            raise ValueError(
                "saturation_max must be between 0 and 255."
            )

        if not 0 <= self.value_min <= 255:
            raise ValueError(
                "value_min must be between 0 and 255."
            )

        if not 0 <= self.value_max <= 255:
            raise ValueError(
                "value_max must be between 0 and 255."
            )

        if self.hue_min > self.hue_max:
            raise ValueError(
                "hue_min must not exceed hue_max."
            )

        if self.saturation_min > self.saturation_max:
            raise ValueError(
                "saturation_min must not exceed saturation_max."
            )

        if self.value_min > self.value_max:
            raise ValueError(
                "value_min must not exceed value_max."
            )

def convert_bgr_to_hsv(
        frame: np.ndarray,
) -> np.ndarray:
    """Convert an OpenCV BGR image into HSV colour space."""

    return cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2HSV,
    )

def create_hsv_mask(
        frame: np.ndarray,
        colour_range: HSVRange,
) -> np.ndarray:
    """Return a binary mask containing pixels within an HSV range."""

    hsv = convert_bgr_to_hsv(frame)

    lower_bound = np.array(
        [
            colour_range.hue_min,
            colour_range.saturation_min,
            colour_range.value_min
        ],
        dtype=np.uint8,
    )

    upper_bound = np.array(
        [colour_range.hue_max,
         colour_range.saturation_max,
         colour_range.value_max
        ],
        dtype=np.uint8,
    )
    return cv2.inRange(
        hsv,
        lower_bound,
        upper_bound
    )