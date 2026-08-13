from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class GroundTruthPose:
    centroid: tuple[float, float]
    angle_degrees: float

@dataclass(frozen=True, slots=True)
class EstimatedPose:
    centroid: tuple[float, float]
    angle_degrees: float

@dataclass(frozen=True, slots=True)
class FrameEvaluation:
    target_present: bool
    detected: bool
    latency_ms: float

    centroid_error_pixels: float | None = None
    orientation_error_degrees: float | None = None


@dataclass(frozen=True, slots=True)
class PerceptionEvaluationReport:
    frame_count: int

    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int

    precision: float | None
    recall: float | None
    false_positive_rate: float | None

    mean_centroid_error_pixels: float | None
    median_centroid_error_pixels: float | None

    mean_orientation_error_degrees: float | None
    median_orientation_error_degrees: float | None

    mean_latency_ms: float
    p95_latency_ms: float