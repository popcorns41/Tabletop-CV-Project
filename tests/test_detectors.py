import numpy as np

from tabletop_vision.evaluation.detectors import create_hsv_detector
from tabletop_vision.perception.colour import HSVRange

import cv2

def test_hsv_detector_detects_target() -> None:
    frame = np.zeros(
        (200, 200 , 3),
        dtype=np.uint8,
    )

    cv2.rectangle(
        frame,
        (50, 75),
        (150, 125),
        (255, 0, 0),
        thickness=-1,
    )

    detector = create_hsv_detector(
        hsv_range=HSVRange(
            hue_min= 100,
            saturation_min= 100,
            value_min= 100,

            hue_max= 140,
            saturation_max=255,
            value_max=255,
        ),
        minimum_area=500.0,
    )

    detection = detector(
        frame
    )

    assert detection is not None
    
    assert detection.centroid == (
        100,
        100,
    )

