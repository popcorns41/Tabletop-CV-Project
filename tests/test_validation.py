from tabletop_vision.dataset.models import (
    DatasetImageMetadata,
    PolygonPoint,
    InstanceAnnotation,
    ImageAnnotation,
    DatasetSplit
)

from tabletop_vision.dataset.validation import (
    validate_dataset
)

def test_valid_dataset_passes(
        tmp_path,
) -> None:
    images_directory = (
        tmp_path / "images"
    )

    images_directory.mkdir()

    for filename in (
        "a.jpg",
        "b.jpg",
        "c.jpg",
    ):
        (
            images_directory / filename
        ).touch()

    metadata = [
        DatasetImageMetadata(
            filename=filename,
            width=1280,
            height=720,
             timestamp="2026-01-01T00:00:00+00:00",
            environment="test",
            session=f"session_{filename}",
            target_present=True,
        )
        for filename in (
            "a.jpg",
            "b.jpg",
            "c.jpg",
        )
    ]

    polygon = (
        PolygonPoint(0.2, 0.2),
        PolygonPoint(0.8,0.2),
        PolygonPoint(0.8,0.8),
        PolygonPoint(0.2,0.8),
    )

    annotations = [
        ImageAnnotation(
            filename=filename,
            instances=(
                InstanceAnnotation(
                    class_name="target_object",
                    polygon=polygon
                ),
            ),
        )
        for filename in (
            "a.jpg",
            "b.jpg",
            "c.jpg",
        )
    ]

    split = DatasetSplit(
        train=("a.jpg",),
        validation=("b.jpg",),
        test=("c.jpg",),
    )

    report = validate_dataset(
        metadata=metadata,
        annotations=annotations,
        split=split,
        images_directory=images_directory,
        allowed_classes={"target_object"},
    )

    assert report.is_valid
    assert report.errors == ()

def test_validator_detects_split_leakage(
    tmp_path,
) -> None:
    images_directory = (
        tmp_path / "images"
    )

    images_directory.mkdir()

    (
        images_directory / "frame.jpg"
    ).touch()

    metadata = [
        DatasetImageMetadata(
            filename="frame.jpg",
            width=1280,
            height=720,
            timestamp="2026-01-01T00:00:00+00:00",
            environment="test",
            session="session_a",
            target_present=False,
        )
    ]

    annotations = [
        ImageAnnotation(
            filename="frame.jpg",
            instances=(),
        )
    ]

    split = DatasetSplit(
        train=("frame.jpg",),
        validation=("frame.jpg",),
        test=(),
    )

    report = validate_dataset(
        metadata=metadata,
        annotations=annotations,
        split=split,
        images_directory=images_directory,
        allowed_classes={"target_object"},
    )

    assert not report.is_valid

    assert any(
        "Split leakage" in error
        for error in report.errors
    )

def test_validator_detects_missing_image(
    tmp_path,
) -> None:
    images_directory = (
        tmp_path / "images"
    )

    images_directory.mkdir()

    metadata = [
        DatasetImageMetadata(
            filename="missing.jpg",
            width=1280,
            height=720,
            timestamp="2026-01-01T00:00:00+00:00",
            environment="test",
            session="session_a",
            target_present=False,
        )
    ]

    annotations = [
        ImageAnnotation(
            filename="missing.jpg",
            instances=(),
        )
    ]

    split = DatasetSplit(
        train=("missing.jpg",),
        validation=(),
        test=(),
    )

    report = validate_dataset(
        metadata=metadata,
        annotations=annotations,
        split=split,
        images_directory=images_directory,
        allowed_classes={"target_object"},
    )

    assert not report.is_valid

    assert any(
        "Metadata references missing image" in error
        for error in report.errors
    )

def test_validator_detects_unknown_class(
    tmp_path,
) -> None:
    images_directory = (
        tmp_path / "images"
    )

    images_directory.mkdir()

    (
        images_directory / "frame.jpg"
    ).touch()

    metadata = [
        DatasetImageMetadata(
            filename="frame.jpg",
            width=1280,
            height=720,
            timestamp="2026-01-01T00:00:00+00:00",
            environment="test",
            session="session_a",
            target_present=True,
        )
    ]

    polygon = (
        PolygonPoint(0.2, 0.2),
        PolygonPoint(0.8, 0.2),
        PolygonPoint(0.8, 0.8),
        PolygonPoint(0.2, 0.8),
    )

    annotations = [
        ImageAnnotation(
            filename="frame.jpg",
            instances=(
                InstanceAnnotation(
                    class_name="mystery_object",
                    polygon=polygon,
                ),
            ),
        )
    ]

    split = DatasetSplit(
        train=("frame.jpg",),
        validation=(),
        test=(),
    )

    report = validate_dataset(
        metadata=metadata,
        annotations=annotations,
        split=split,
        images_directory=images_directory,
        allowed_classes={"target_object"},
    )

    assert not report.is_valid

    assert any(
        "Unknown class 'mystery_object'" in error
        for error in report.errors
    )

def test_validator_detects_duplicate_annotation(
    tmp_path,
) -> None:
    images_directory = (
        tmp_path / "images"
    )

    images_directory.mkdir()

    (
        images_directory / "frame.jpg"
    ).touch()

    metadata = [
        DatasetImageMetadata(
            filename="frame.jpg",
            width=1280,
            height=720,
            timestamp="2026-01-01T00:00:00+00:00",
            environment="test",
            session="session_a",
            target_present=True,
        )
    ]

    polygon = (
        PolygonPoint(0.2, 0.2),
        PolygonPoint(0.8, 0.2),
        PolygonPoint(0.8, 0.8),
        PolygonPoint(0.2, 0.8),
    )

    annotation = ImageAnnotation(
        filename="frame.jpg",
        instances=(
            InstanceAnnotation(
                class_name="target_object",
                polygon=polygon,
            ),
        ),
    )

    annotations = [
        annotation,
        annotation,
    ]

    split = DatasetSplit(
        train=("frame.jpg",),
        validation=(),
        test=(),
    )

    report = validate_dataset(
        metadata=metadata,
        annotations=annotations,
        split=split,
        images_directory=images_directory,
        allowed_classes={"target_object"},
    )

    assert not report.is_valid

    assert any(
        "Duplicate annotation entry: frame.jpg" in error
        for error in report.errors
    )