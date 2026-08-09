import pytest

from tabletop_vision.dataset import (
    AnnotationWriter,
    ImageAnnotation,
    InstanceAnnotation,
    PolygonPoint,
    load_annotations,
)

def test_polygon_point_rejects_invalid_coordinates() -> None:
    with pytest.raises(ValueError):
        PolygonPoint(
            x=1.2,
            y=0.5,
        )

def test_instance_annotation_requires_three_points() -> None:
    with pytest.raises(ValueError):
        InstanceAnnotation(
            class_name="target_object",
            polygon=(
                PolygonPoint(0.1, 0.1),
                PolygonPoint(0.9, 0.9),
            ),
        )


def test_annotation_round_trip(
    tmp_path,
) -> None:
    output_path = (
        tmp_path / "annotations.jsonl"
    )

    annotation = ImageAnnotation(
        filename="frame_000001.jpg",
        instances=(
            InstanceAnnotation(
                class_name="target_object",
                polygon=(
                    PolygonPoint(0.2, 0.3),
                    PolygonPoint(0.7, 0.3),
                    PolygonPoint(0.8, 0.7),
                    PolygonPoint(0.2, 0.7),
                ),
            ),
        ),
    )

    writer = AnnotationWriter(
        output_path
    )

    writer.append(
        annotation
    )

    loaded = load_annotations(
        output_path
    )

    # Dataclass is frozen, so the assert below checks the 
    # whole nested structure
    assert loaded == [
        annotation
    ]

def test_annotation_supports_negative_image(
    tmp_path,
) -> None:
    output_path = (
        tmp_path / "annotations.jsonl"
    )

    annotation = ImageAnnotation(
        filename="frame_000002.jpg",
        instances=(),
    )

    writer = AnnotationWriter(
        output_path
    )

    writer.append(
        annotation
    )

    loaded = load_annotations(
        output_path
    )

    assert loaded == [
        annotation
    ]