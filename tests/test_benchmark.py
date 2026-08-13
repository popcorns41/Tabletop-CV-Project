import numpy as np

from tabletop_vision.evaluation.benchmark import (
    BenchmarkSample,
    benchmark_detector,
)

from tabletop_vision.evaluation.models import (
    GroundTruthPose,
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

def _create_detection() -> ObjectDetection:
    polygon = np.array(
        [
            [40.0, 45.0],
            [60.0, 45.0],
            [60.0, 55.0],
            [40.0, 55.0],
        ],
        dtype=np.float32,
    )

    contour = polygon_to_contour(
        polygon
    )

    centroid = contour_centroid(
        contour
    )

    assert centroid is not None

    return ObjectDetection(
        contour=contour,
        centroid=centroid,
        area_pixels_squared=(
            contour_area(contour)
        ),
        rotated_rectangle=(
            contour_rotated_rectangle(contour)
        ),
    )


def test_benchmark_detector() -> None:
    frame = np.zeros(
        (100, 100, 3),
        dtype=np.uint8,
    )

    detection = _create_detection()

    def fake_detector(
            _:np.ndarray,
    ) -> ObjectDetection | None:
        return detection

    samples = (
        BenchmarkSample(
            frame=frame,
            ground_truth=GroundTruthPose(
                centroid=(50.0, 50.0),
                angle_degrees=0.0,
            ),
        ),
    )

    result = benchmark_detector(
        detector_name="Fake Detector",
        detector=fake_detector,
        samples=samples,
    )

    assert result.detector_name == (
        "Fake Detector"
    )

    assert result.report.frame_count == 1
    assert result.report.true_positives == 1
    assert result.report.false_positives == 0

    assert result.report.precision == 1.0
    assert result.report.recall == 1.0

    assert (
        result.report.mean_centroid_error_pixels
        == 0.0
    )

    assert (
        result.report.mean_orientation_error_degrees
        == 0.0
    )

    assert (
        result.report.mean_latency_ms
        >= 0.0
    )

def test_benchmark_records_false_positive(   
) -> None:
    frame = np.zeros(
        (100, 100, 3),
        dtype=np.uint8,
    )

    detection = _create_detection()

    def fake_detector(
        _: np.ndarray,
    ) -> ObjectDetection | None:
        return detection

    samples = (
        BenchmarkSample(
            frame=frame,
            ground_truth=None,
        ),
    )


    result = benchmark_detector(
        detector_name="Fake Detector",
        detector=fake_detector,
        samples=samples,
    )

    assert result.report.true_positives == 0
    assert result.report.false_positives == 1
    assert result.report.true_negatives == 0

