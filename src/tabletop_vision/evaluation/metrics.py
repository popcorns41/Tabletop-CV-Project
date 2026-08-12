from __future__ import annotations

from tabletop_vision.evaluation.models import (
    EstimatedPose,
    FrameEvaluation,
    GroundTruthPose,
)

from tabletop_vision.perception.models import(
    ObjectDetection,
)

from collections.abc import Sequence

import numpy as np

from tabletop_vision.evaluation.models import (
    FrameEvaluation,
    PerceptionEvaluationReport,
)

import math

def aggregate_evaluations(
        evaluations: Sequence[
            FrameEvaluation
        ],
) -> PerceptionEvaluationReport:
    if not evaluations:
        raise ValueError(
            "At least one frame evaluation "
            "is required."
        )

    detected = [
        evaluation
        for evaluation in evaluations
        if evaluation.detected
    ]

    centroid_errors = [
        evaluation.centroid_error_pixels
        for evaluation in detected
        if (
            evaluation.centroid_error_pixels
            is not None
        )
    ]

    orientation_errors = [
        evaluation.orientation_error_degrees
        for evaluation in detected
        if (
            evaluation.orientation_error_degrees
            is not None
        )
    ]

    latencies = [
        evaluation.latency_ms
        for evaluation in evaluations
    ]

    detection_rate = (
        len(detected)
        / len(evaluations)
    )

    return PerceptionEvaluationReport(
        frame_count=len(evaluations),

        detection_rate=detection_rate,

        mean_centroid_error_pixels=(
            float(np.mean(centroid_errors))
            if centroid_errors
            else None
        ),

        median_centroid_error_pixels=(
            float(np.median(centroid_errors))
            if centroid_errors
            else None
        ),

        mean_orientation_error_degrees = (
            float(np.mean(orientation_errors))
            if orientation_errors
            else None
        ),

        median_orientation_error_degrees=(
            float(np.median(orientation_errors))
            if orientation_errors
            else None
        ),

        mean_latency_ms=float(
            np.mean(latencies)
        ),

        p95_latency_ms=float(
            np.percentile(
                latencies,
                95,
            )
        ),
    )



def evaluate_detection(
        detection: ObjectDetection | None,
        prediction: EstimatedPose,
        ground_truth: GroundTruthPose,
        latency_ms: float,
) -> FrameEvaluation:
    if latency_ms < 0.0:
        raise ValueError(
            "latency_ms must not be negative"
        )

    if detection is None:
        return FrameEvaluation(
            detected = False,
            latency_ms=latency_ms,
        )


    pos_err_px = positional_error_pixels(
        prediction.centroid,
        ground_truth.centroid,
    )

    orient_err_deg = orientation_error_degrees(
        prediction.angle_degrees,
        ground_truth.angle_degrees,
    )

    return FrameEvaluation(
        centroid_error_pixels=pos_err_px,
        orientation_error_degrees=orient_err_deg,
        latency_ms=latency_ms,
        detected=True,
    )


def positional_error_pixels(
        predicted: tuple[float, float],
        ground_truth: tuple[float, float],
) -> float:
    dx = (
        predicted[0]
        - ground_truth[0]
    )

    dy = (
        predicted[1]
        - ground_truth[1]
    )

    return math.hypot(
        dx,
        dy,
    )

def orientation_error_degrees(
        predicted: float,
        ground_truth: float,
) -> float:
    difference = abs (
        predicted
        - ground_truth
    ) % 180.0

    return min(difference, 180.0 - difference,)

