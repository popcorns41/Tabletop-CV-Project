

from tabletop_vision.evaluation.benchmark import FrameDetector
from tabletop_vision.learned.detection import (
    prediction_to_object_detection, 
    select_target_prediction
) 
from tabletop_vision.learned.segmenter import InstanceSegmenter
from tabletop_vision.perception.colour import (
    HSVRange, 
    create_hsv_mask
)
from tabletop_vision.perception.detection import detect_largest_object
from tabletop_vision.perception.models import ObjectDetection

import numpy as np

#function factories for different types of detectors

def create_hsv_detector(
        hsv_range: HSVRange,
        minimum_area: float,
) -> FrameDetector:
    if minimum_area <= 0.0:
        raise ValueError(
            "minimum_area must be positive."
        )

    def detect(
            frame: np.ndarray,
    ) -> ObjectDetection | None:
        mask = create_hsv_mask(
            frame,
            hsv_range,
        )

        return detect_largest_object(
            mask,
            minimum_area=minimum_area,
        )

    return detect

def create_learned_detector(
    segmenter: InstanceSegmenter,
    class_name: str,
) -> FrameDetector:
    if not class_name.strip():
        raise ValueError(
            "class_name must not be empty."
        )

    def detect(
            frame: np.ndarray,
    ) -> ObjectDetection | None:
        predictions = segmenter.predict(
            frame
        )

        target = select_target_prediction(
            predictions,
            class_name=class_name,
        )

        if target is None:
            return None

        return prediction_to_object_detection(
            target
        )
    
    return detect
    