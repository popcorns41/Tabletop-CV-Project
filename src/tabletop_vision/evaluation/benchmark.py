from __future__ import annotations

import time
from collections.abc import (
    Callable,
    Sequence,
)

from dataclasses import dataclass

import numpy as np

from tabletop_vision.evaluation.metrics import (
    aggregate_evaluations,
    evaluate_detection,
)

from tabletop_vision.evaluation.models import (
    FrameEvaluation,
    GroundTruthPose,
    PerceptionEvaluationReport,
)

from tabletop_vision.perception.models import (
    ObjectDetection,
)


#Frame detector is a generic type alias that supports
#any function that accepts a frame and outputs
# -> ObjectDetection | None
FrameDetector = Callable[
    [np.ndarray],
    ObjectDetection | None,
]

@dataclass(frozen=True, slots=True)
class BenchmarkSample:
    frame: np.ndarray
    ground_truth: GroundTruthPose | None

@dataclass(frozen=True, slots=True)
class DetectorBenchmarkResult:
    detector_name: str
    evaluations: tuple[
        FrameEvaluation,
        ...
    ]

    report: PerceptionEvaluationReport

def benchmark_detector(
        detector_name: str,
        detector: FrameDetector,
        samples: Sequence[
            BenchmarkSample
        ],
) -> DetectorBenchmarkResult:
    if not detector_name.strip():
        raise ValueError(
            "detector_name must not be empty."
        )

    if not samples:
        raise ValueError(
            "At least one benchmark sample "
            "is required."
        )

    evaluations: list[
        FrameEvaluation
    ] = []

    for sample in samples:
        frame = sample.frame.copy()

        start = time.perf_counter_ns()

        detection = detector(
            frame
        )

        end = time.perf_counter_ns()

        latency_ms = (
            end - start
        ) / 1_000_000.0

        evaluation = evaluate_detection(
            detection=detection,
            ground_truth=(sample.ground_truth),
            latency_ms=latency_ms,
        )

        evaluations.append(evaluation)

    report = aggregate_evaluations(
        evaluations
    )

    return DetectorBenchmarkResult(
        detector_name=detector_name,
        evaluations=tuple(evaluations),
        report=report,
    )

def benchmark_detectors(
        detectors: dict[
            str,
            FrameDetector,
        ],
        samples: Sequence[
            BenchmarkSample
        ],
) -> tuple[DetectorBenchmarkResult,...]:
    if not detectors:
        raise ValueError(
            "At least one detector is required."
        )

    results = [
        benchmark_detector(
            detector_name=name,
            detector=detector,
            samples=samples,
        )
        for name, detector
        in detectors.items()
    ]

    return tuple(results)