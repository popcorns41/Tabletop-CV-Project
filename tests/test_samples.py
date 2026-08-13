from pathlib import Path

import cv2
import numpy as np
import pytest

from tabletop_vision.dataset.models import (
    DatasetImageMetadata,
    ImageAnnotation,
    InstanceAnnotation,
    PolygonPoint,
)
from tabletop_vision.evaluation.samples import (
    annotation_to_ground_truth_pose,
    create_benchmark_sample,
    create_benchmark_samples,
)


def test_annotation_to_ground_truth_pose() -> None:
    metadata = DatasetImageMetadata(
        filename="frame.jpg",
        width=100,
        height=100,
        timestamp="2026-01-01T00:00:00+00:00",
        environment="test",
        session="session_a",
        target_present=True,
    )

    annotation = ImageAnnotation(
        filename="frame.jpg",
        instances=(
            InstanceAnnotation(
                class_name="target_object",
                polygon=(
                    PolygonPoint(0.2, 0.4),
                    PolygonPoint(0.8, 0.4),
                    PolygonPoint(0.8, 0.6),
                    PolygonPoint(0.2, 0.6),
                ),
            ),
        ),
    )

    pose = annotation_to_ground_truth_pose(
        annotation=annotation,
        metadata=metadata,
        class_name="target_object",
    )

    assert pose is not None

    assert pose.centroid == (
        50.0,
        50.0,
    )

    assert pose.angle_degrees == 0.0

def test_annotation_without_target_returns_none(
) -> None:
    metadata = DatasetImageMetadata(
        filename="frame.jpg",
        width=100,
        height=100,
        timestamp="2026-01-01T00:00:00+00:00",
        environment="test",
        session="session_a",
        target_present=False,
    )

    annotation = ImageAnnotation(
        filename="frame.jpg",
        instances=(),
    )

    pose = annotation_to_ground_truth_pose(
        annotation=annotation,
        metadata=metadata,
        class_name="target_object",
    )

    assert pose is None


def test_annotation_rejects_multiple_targets() -> None:
    metadata = DatasetImageMetadata(
        filename="frame.jpg",
        width=100,
        height=100,
        timestamp="2026-01-01T00:00:00+00:00",
        environment="test",
        session="session_a",
        target_present=True,
    )

    polygon = (
        PolygonPoint(0.2, 0.2),
        PolygonPoint(0.4, 0.2),
        PolygonPoint(0.4, 0.4),
        PolygonPoint(0.2, 0.4),
    )

    annotation = ImageAnnotation(
        filename="frame.jpg",
        instances=(
            InstanceAnnotation(
                class_name="target_object",
                polygon=polygon,
            ),
            InstanceAnnotation(
                class_name="target_object",
                polygon=polygon,
            ),
        ),
    )

    with pytest.raises(ValueError):
        annotation_to_ground_truth_pose(
            annotation=annotation,
            metadata=metadata,
            class_name="target_object",
        )

def test_annotation_ignores_other_classes() -> None:
    metadata = DatasetImageMetadata(
        filename="frame.jpg",
        width=100,
        height=100,
        timestamp="2026-01-01T00:00:00+00:00",
        environment="test",
        session="session_a",
        target_present=True,
    )

    annotation = ImageAnnotation(
        filename="frame.jpg",
        instances=(
            InstanceAnnotation(
                class_name="other_object",
                polygon=(
                    PolygonPoint(0.2, 0.2),
                    PolygonPoint(0.4, 0.2),
                    PolygonPoint(0.4, 0.4),
                ),
            ),
        ),
    )

    pose = annotation_to_ground_truth_pose(
        annotation=annotation,
        metadata=metadata,
        class_name="target_object",
    )

    assert pose is None

def test_create_benchmark_sample(
    tmp_path: Path,
) -> None:
    images_directory = (
        tmp_path / "images"
    )

    images_directory.mkdir()

    frame = np.zeros(
        (100, 100, 3),
        dtype=np.uint8,
    )

    cv2.imwrite(
        str(
            images_directory
            / "frame.jpg"
        ),
        frame,
    )

    metadata = DatasetImageMetadata(
        filename="frame.jpg",
        width=100,
        height=100,
        timestamp="2026-01-01T00:00:00+00:00",
        environment="test",
        session="session_a",
        target_present=True,
    )

    annotation = ImageAnnotation(
        filename="frame.jpg",
        instances=(
            InstanceAnnotation(
                class_name="target_object",
                polygon=(
                    PolygonPoint(0.2, 0.4),
                    PolygonPoint(0.8, 0.4),
                    PolygonPoint(0.8, 0.6),
                    PolygonPoint(0.2, 0.6),
                ),
            ),
        ),
    )

    sample = create_benchmark_sample(
        filename="frame.jpg",
        images_directory=images_directory,
        metadata=metadata,
        annotation=annotation,
        class_name="target_object",
    )

    assert sample.frame.shape == (
        100,
        100,
        3,
    )

    assert sample.ground_truth is not None
    assert sample.ground_truth.centroid == (
        50.0,
        50.0,
    )

def test_create_benchmark_sample_rejects_missing_image(
    tmp_path: Path,
) -> None:
    images_directory = (
        tmp_path / "images"
    )

    images_directory.mkdir()

    metadata = DatasetImageMetadata(
        filename="missing.jpg",
        width=100,
        height=100,
        timestamp="2026-01-01T00:00:00+00:00",
        environment="test",
        session="session_a",
        target_present=False,
    )

    annotation = ImageAnnotation(
        filename="missing.jpg",
        instances=(),
    )

    with pytest.raises(ValueError):
        create_benchmark_sample(
            filename="missing.jpg",
            images_directory=images_directory,
            metadata=metadata,
            annotation=annotation,
            class_name="target_object",
        )

def test_create_benchmark_sample_rejects_dimension_mismatch(
    tmp_path: Path,
) -> None:
    images_directory = (
        tmp_path / "images"
    )

    images_directory.mkdir()

    frame = np.zeros(
        (50, 50, 3),
        dtype=np.uint8,
    )

    cv2.imwrite(
        str(
            images_directory
            / "frame.jpg"
        ),
        frame,
    )

    metadata = DatasetImageMetadata(
        filename="frame.jpg",
        width=100,
        height=100,
        timestamp="2026-01-01T00:00:00+00:00",
        environment="test",
        session="session_a",
        target_present=False,
    )

    annotation = ImageAnnotation(
        filename="frame.jpg",
        instances=(),
    )

    with pytest.raises(ValueError):
        create_benchmark_sample(
            filename="frame.jpg",
            images_directory=images_directory,
            metadata=metadata,
            annotation=annotation,
            class_name="target_object",
        )

def test_create_benchmark_samples(
    tmp_path: Path,
) -> None:
    images_directory = (
        tmp_path / "images"
    )

    images_directory.mkdir()

    for filename in (
        "positive.jpg",
        "negative.jpg",
    ):
        frame = np.zeros(
            (100, 100, 3),
            dtype=np.uint8,
        )

        cv2.imwrite(
            str(
                images_directory
                / filename
            ),
            frame,
        )

    metadata = (
        DatasetImageMetadata(
            filename="positive.jpg",
            width=100,
            height=100,
            timestamp="2026-01-01T00:00:00+00:00",
            environment="test",
            session="session_a",
            target_present=True,
        ),
        DatasetImageMetadata(
            filename="negative.jpg",
            width=100,
            height=100,
            timestamp="2026-01-01T00:00:01+00:00",
            environment="test",
            session="session_a",
            target_present=False,
        ),
    )

    annotations = (
        ImageAnnotation(
            filename="positive.jpg",
            instances=(
                InstanceAnnotation(
                    class_name="target_object",
                    polygon=(
                        PolygonPoint(0.2, 0.4),
                        PolygonPoint(0.8, 0.4),
                        PolygonPoint(0.8, 0.6),
                        PolygonPoint(0.2, 0.6),
                    ),
                ),
            ),
        ),
        ImageAnnotation(
            filename="negative.jpg",
            instances=(),
        ),
    )

    samples = create_benchmark_samples(
        filenames=(
            "positive.jpg",
            "negative.jpg",
        ),
        images_directory=images_directory,
        metadata=metadata,
        annotations=annotations,
        class_name="target_object",
    )

    assert len(samples) == 2

    assert samples[0].ground_truth is not None
    assert samples[1].ground_truth is None

def test_create_benchmark_samples_rejects_missing_metadata(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError):
        create_benchmark_samples(
            filenames=("frame.jpg",),
            images_directory=tmp_path,
            metadata=(),
            annotations=(
                ImageAnnotation(
                    filename="frame.jpg",
                    instances=(),
                ),
            ),
            class_name="target_object",
        )

def test_create_benchmark_samples_rejects_missing_annotation(
    tmp_path: Path,
) -> None:
    metadata = (
        DatasetImageMetadata(
            filename="frame.jpg",
            width=100,
            height=100,
            timestamp="2026-01-01T00:00:00+00:00",
            environment="test",
            session="session_a",
            target_present=False,
        ),
    )

    with pytest.raises(ValueError):
        create_benchmark_samples(
            filenames=("frame.jpg",),
            images_directory=tmp_path,
            metadata=metadata,
            annotations=(),
            class_name="target_object",
        )

def test_create_benchmark_samples_rejects_duplicate_metadata(
    tmp_path: Path,
) -> None:
    metadata_item = DatasetImageMetadata(
        filename="frame.jpg",
        width=100,
        height=100,
        timestamp="2026-01-01T00:00:00+00:00",
        environment="test",
        session="session_a",
        target_present=False,
    )

    with pytest.raises(ValueError):
        create_benchmark_samples(
            filenames=("frame.jpg",),
            images_directory=tmp_path,
            metadata=(
                metadata_item,
                metadata_item,
            ),
            annotations=(),
            class_name="target_object",
        )