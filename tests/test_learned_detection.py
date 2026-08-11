import numpy as np

from tabletop_vision.learned import (
    SegmentationPrediction,
)
from tabletop_vision.learned.detection import (
    prediction_to_object_detection,
    select_target_prediction
)


def test_prediction_converts_to_object_detection(
) -> None:
    prediction = SegmentationPrediction(
        class_id=0,
        class_name="target_object",
        confidence=0.95,
        mask=np.zeros(
            (100, 100),
            dtype=np.uint8,
        ),
        polygon=np.array(
            [
                [20.0, 30.0],
                [80.0, 30.0],
                [80.0, 70.0],
                [20.0, 70.0],
            ],
            dtype=np.float32,
        ),
    )

    detection = (
        prediction_to_object_detection(
            prediction
        )
    )

    assert detection is not None

    assert detection.centroid == (
        50,
        50,
    )

    assert (
        detection.area_pixels_squared
        == 2400.0
    )

    assert (
        detection.rotated_rectangle.width
        == 60.0
    )

    assert (
        detection.rotated_rectangle.height
        == 40.0
    )

def test_select_target_prediction_uses_highest_confidence(
) -> None:
    polygon = np.array(
        [
            [0.0, 0.0],
            [10.0, 0.0],
            [10.0, 10.0],
        ],
        dtype=np.float32,
    )

    predictions = [
        SegmentationPrediction(
            class_id=0,
            class_name="target_object",
            confidence=0.60,
            mask=np.zeros(
                (10, 10),
                dtype=np.uint8,
            ),
            polygon=polygon,
        ),
        SegmentationPrediction(
            class_id=0,
            class_name="target_object",
            confidence=0.92,
            mask=np.zeros(
                (10, 10),
                dtype=np.uint8,
            ),
            polygon=polygon,
        ),
    ]

    selected = (
        select_target_prediction(
            predictions,
            "target_object",
        )
    )

    assert selected is not None
    assert selected.confidence == 0.92

