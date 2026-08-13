from collections.abc import Sequence

from tabletop_vision.evaluation.benchmark import (
    DetectorBenchmarkResult,
)

import json
from pathlib import Path

def benchmark_results_to_dict(
    results: Sequence[
        DetectorBenchmarkResult
    ],
) -> dict[str, object]:
    if not results:
        raise ValueError(
            "At least one benchmark result "
            "is required."
        )

    detectors: dict[
        str,
        object,
    ] = {}

    for result in results:
        report = result.report

        detectors[
            result.detector_name
        ] = {
            "frame_count": (
                report.frame_count
            ),
            "true_positives": (
                report.true_positives
            ),
            "false_positives": (
                report.false_positives
            ),
            "true_negatives": (
                report.true_negatives
            ),
            "false_negatives": (
                report.false_negatives
            ),
            "precision": (
                report.precision
            ),
            "recall": (
                report.recall
            ),
            "false_positive_rate": (
                report.false_positive_rate
            ),
            "mean_centroid_error_pixels": (
                report
                .mean_centroid_error_pixels
            ),
            "median_centroid_error_pixels": (
                report
                .median_centroid_error_pixels
            ),
            "mean_orientation_error_degrees": (
                report
                .mean_orientation_error_degrees
            ),
            "median_orientation_error_degrees": (
                report
                .median_orientation_error_degrees
            ),
            "mean_latency_ms": (
                report.mean_latency_ms
            ),
            "p95_latency_ms": (
                report.p95_latency_ms
            ),
        }

    return {
        "detectors": detectors,
    }

def save_benchmark_results(
        results: Sequence[
            DetectorBenchmarkResult
        ],
        output_path: Path,
) -> None:
    payload = benchmark_results_to_dict(
        results
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            payload,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

def format_benchmark_report(
    results: Sequence[
        DetectorBenchmarkResult
    ],
) -> str:
    if not results:
        raise ValueError(
            "At least one benchmark result "
            "is required."
        )

    label_width = 32

    value_width = max(
        14,
        max(
            len(result.detector_name) + 2
            for result in results
        ),
    )

    lines = [
        "Perception Benchmark",
        "",
    ]

    lines.append(
        _format_row(
            "Metric",
            [
                result.detector_name
                for result in results
            ],
            label_width,
            value_width,
        )
    )

    lines.append(
        "-" * (
            label_width
            + value_width * len(results)
        )
    )

    reports = [
        result.report
        for result in results
    ]

    lines.append(
        _format_row(
            "Frames",
            [
                str(report.frame_count)
                for report in reports
            ],
            label_width,
            value_width,
        )
    )

    lines.append(
        _format_row(
            "True positives",
            [
                str(report.true_positives)
                for report in reports
            ],
            label_width,
            value_width,
        )
    )

    lines.append(
        _format_row(
            "False positives",
            [
                str(report.false_positives)
                for report in reports
            ],
            label_width,
            value_width,
        )
    )

    lines.append(
        _format_row(
            "True negatives",
            [
                str(report.true_negatives)
                for report in reports
            ],
            label_width,
            value_width,
        )
    )

    lines.append(
        _format_row(
            "False negatives",
            [
                str(report.false_negatives)
                for report in reports
            ],
            label_width,
            value_width,
        )
    )

    lines.append(
        _format_row(
            "Precision",
            [
                _format_percentage(
                    report.precision
                )
                for report in reports
            ],
            label_width,
            value_width,
        )
    )

    lines.append(
        _format_row(
            "Recall",
            [
                _format_percentage(
                    report.recall
                )
                for report in reports
            ],
            label_width,
            value_width,
        )
    )

    lines.append(
        _format_row(
            "False-positive rate",
            [
                _format_percentage(
                    report.false_positive_rate
                )
                for report in reports
            ],
            label_width,
            value_width,
        )
    )

    lines.append(
        _format_row(
            "Median centroid error",
            [
                _format_number(
                    report
                    .median_centroid_error_pixels,
                    " px",
                )
                for report in reports
            ],
            label_width,
            value_width,
        )
    )

    lines.append(
        _format_row(
            "Median orientation error",
            [
                _format_number(
                    report
                    .median_orientation_error_degrees,
                    " deg",
                )
                for report in reports
            ],
            label_width,
            value_width,
        )
    )

    lines.append(
        _format_row(
            "Mean latency",
            [
                _format_number(
                    report.mean_latency_ms,
                    " ms",
                )
                for report in reports
            ],
            label_width,
            value_width,
        )
    )

    lines.append(
        _format_row(
            "P95 latency",
            [
                _format_number(
                    report.p95_latency_ms,
                    " ms",
                )
                for report in reports
            ],
            label_width,
            value_width,
        )
    )

    return "\n".join(lines)

def _format_number(
        value: float | None,
        suffix: str = "",
        decimal_places: int = 2,
) -> str:
    if value is None:
        return "N/A"

    return (
        f"{value:.{decimal_places}f}"
        f"{suffix}"
    )

def _format_percentage(
    value: float | None,
) -> str:
    if value is None:
        return "N/A"

    return f"{value * 100.0:.2f}%"

def _format_row(
        label: str,
        values: list[str],
        label_width: int,
        value_width: int,
) -> str:
    columns = [
        f"{label:<{label_width}}"
    ]

    columns.extend(
        f"{value:>{value_width}}"
        for value in values
    )

    return "".join(columns)

