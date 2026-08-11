from __future__ import annotations

from tabletop_vision.learned.models import (
    SegmentationPrediction,
)

from tabletop_vision.perception.contours import (
    contour_area,
    contour_centroid,
    contour_rotated_rectangle,
    polygon_to_contour,
)

from tabletop_vision.perception.models import (
    ObjectDetection,
)

def prediction_to_object_detection(
        prediction: SegmentationPrediction,
) -> ObjectDetection | None:
    """Convert a segmentation prediction to an object detection."""
    contour = polygon_to_contour(
        prediction.polygon
    )

    centroid = contour_centroid(
        contour
    )

    if centroid is None:
        return None

    area = contour_area(
        contour
    )

    rotated_rectangle = (
        contour_rotated_rectangle(
            contour
        )
    )

    return ObjectDetection(
        contour=contour,
        centroid=centroid,
        area_pixels_squared=area,
        rotated_rectangle=rotated_rectangle,
    )

def select_target_prediction(
        predictions: list[
            SegmentationPrediction
        ],
        class_name: str,
) -> SegmentationPrediction | None:
    candidates = [
        prediction
        for prediction in predictions
        if prediction.class_name
        == class_name
    ]

    if not candidates:
        return None

    return max(
        candidates,
        key=lambda prediction: (
            prediction.confidence
        ),
    )