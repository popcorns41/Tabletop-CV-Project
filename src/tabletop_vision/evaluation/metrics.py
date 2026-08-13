from __future__ import annotations

from tabletop_vision.evaluation.models import (
    GroundTruthPose,
    FrameEvaluation,
    PerceptionEvaluationReport,
)



from tabletop_vision.perception.models import(
    ObjectDetection,
)

from collections.abc import Sequence

import numpy as np

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

    true_positives = sum(
        evaluation.target_present
        and evaluation.detected
        for evaluation in evaluations
    )

    false_positives = sum(
        not evaluation.target_present
        and evaluation.detected
        for evaluation in evaluations
    )

    true_negatives = sum(
        not evaluation.target_present
        and not evaluation.detected
        for evaluation in evaluations
    )

    false_negatives = sum(
        evaluation.target_present
        and not evaluation.detected
        for evaluation in evaluations
    )

    centroid_errors = [
        evaluation.centroid_error_pixels
        for evaluation in evaluations
        if (
            evaluation.centroid_error_pixels
            is not None
        )
    ]

    orientation_errors = [
        evaluation.orientation_error_degrees
        for evaluation in evaluations
        if (
            evaluation.orientation_error_degrees
            is not None
        )
    ]

    latencies = [
        evaluation.latency_ms
        for evaluation in evaluations
    ]

    predicted_positive_count = (
        true_positives
        + false_positives
    )

    actual_positive_count = (
        true_positives
        + false_negatives
    )

    actual_negative_count = (
        true_negatives
        + false_positives
    )

    precision = (
        true_positives
        / predicted_positive_count
        if predicted_positive_count
        else None
    )

    recall = (
        true_positives
        / actual_positive_count
        if actual_positive_count
        else None
    )

    false_positive_rate = (
        false_positives
        / actual_negative_count
        if actual_negative_count
        else None
    )

    return PerceptionEvaluationReport(
        frame_count=len(evaluations),

        true_positives=true_positives,
        false_positives=false_positives,
        true_negatives=true_negatives,
        false_negatives=false_negatives,

        precision=precision,
        recall=recall,
        false_positive_rate=(
            false_positive_rate
        ),

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

        mean_orientation_error_degrees=(
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
    ground_truth: GroundTruthPose | None,
    latency_ms: float,
) -> FrameEvaluation:
    if latency_ms < 0:
        raise ValueError(
            "latency_ms cannot be negative."
        )

    target_present = (
        ground_truth is not None
    )

    if (
        detection is None
        or ground_truth is None
    ):
        return FrameEvaluation(
            target_present=target_present,
            detected=detection is not None,
            latency_ms=latency_ms,
        )

    centroid_error = positional_error_pixels(
        predicted=(
            float(detection.centroid[0]),
            float(detection.centroid[1]),
        ),
        ground_truth=ground_truth.centroid,
    )

    orientation_error = (
        orientation_error_degrees(
            predicted=(
                detection
                .rotated_rectangle
                .angle_degrees
            ),
            ground_truth=(
                ground_truth.angle_degrees
            ),
        )
    )

    return FrameEvaluation(
        target_present=True,
        detected=True,
        latency_ms=latency_ms,
        centroid_error_pixels=(
            centroid_error
        ),
        orientation_error_degrees=(
            orientation_error
        ),
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

