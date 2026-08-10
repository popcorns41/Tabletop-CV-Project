from pathlib import Path

import yaml

from tabletop_vision.dataset import (
    DatasetSplit,
    ImageAnnotation,
    InstanceAnnotation,
    PolygonPoint,
)

from tabletop_vision.learned import (
    export_ultralytics_dataset,
)

def test_export_creates_yolo_segmentation_label(
    tmp_path,
) -> None:
    source_images = (
        tmp_path / "source"
    )

    source_images.mkdir()

    (
        source_images / "frame.jpg"
    ).touch()

    annotation = ImageAnnotation(
        filename="frame.jpg",
        instances=(
            InstanceAnnotation(
                class_name="target_object",
                polygon=(
                    PolygonPoint(0.2, 0.3),
                    PolygonPoint(0.7, 0.3),
                    PolygonPoint(0.7, 0.8),
                ),
            ),
        ),
    )

    split = DatasetSplit(
        train=("frame.jpg",),
        validation=(),
        test=(),
    )

    output = (
        tmp_path / "export"
    )

    export_ultralytics_dataset(
        source_images_directory=source_images,
        annotations=(annotation,),
        split=split,
        output_directory=output,
        class_names=("target_object",),
    )

    label = (
        output
        / "labels"
        / "train"
        / "frame.txt"
    )

    assert label.exists()

    assert label.read_text(
        encoding="utf-8"
    ).strip() == (
        "0 "
        "0.200000 0.300000 "
        "0.700000 0.300000 "
        "0.700000 0.800000"
    )


def test_export_creates_empty_label_for_negative(
    tmp_path,
) -> None:
    source_images = (
        tmp_path / "source"
    )

    source_images.mkdir()

    (
        source_images / "negative.jpg"
    ).touch()

    annotation = ImageAnnotation(
        filename="negative.jpg",
        instances=(),
    )

    split = DatasetSplit(
        train=("negative.jpg",),
        validation=(),
        test=(),
    )

    output = (
        tmp_path / "export"
    )

    export_ultralytics_dataset(
        source_images_directory=source_images,
        annotations=(annotation,),
        split=split,
        output_directory=output,
        class_names=("target_object",),
    )

    label = (
        output
        / "labels"
        / "train"
        / "negative.txt"
    )

    assert label.exists()

    assert label.read_text(
        encoding="utf-8"
    ) == ""

def test_export_creates_dataset_yaml(
    tmp_path,
) -> None:
    source_images = (
        tmp_path / "source"
    )

    source_images.mkdir()

    output = (
        tmp_path / "export"
    )

    yaml_path = export_ultralytics_dataset(
        source_images_directory=source_images,
        annotations=(),
        split=DatasetSplit(
            train=(),
            validation=(),
            test=(),
        ),
        output_directory=output,
        class_names=(
            "target_object",
            "second_object",
        ),
    )

    payload = yaml.safe_load(
        yaml_path.read_text(
            encoding="utf-8"
        )
    )

    assert payload["train"] == (
        "images/train"
    )

    assert payload["val"] == (
        "images/val"
    )

    assert payload["test"] == (
        "images/test"
    )

    assert payload["names"] == {
        0: "target_object",
        1: "second_object",
    }