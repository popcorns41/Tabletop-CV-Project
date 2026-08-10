from __future__ import annotations

import shutil
from collections.abc import Sequence
from pathlib import Path

import yaml

from tabletop_vision.dataset import (
    DatasetSplit,
    ImageAnnotation,
    InstanceAnnotation,
)

def export_ultralytics_dataset(
        source_images_directory: Path,
        annotations: Sequence[ImageAnnotation],
        split: DatasetSplit,
        output_directory: Path,
        class_names: Sequence[str],
) -> Path:
    """Export our internal dataset into Ultralytics segmentation format."""

    if not class_names:
        raise ValueError(
            "At least one class name is required."
        )

    if len(set(class_names)) != len(class_names):
        raise ValueError(
            "Class names must be unique."
        )

    class_to_index = {
        name: index
        for index, name 
        in enumerate(class_names)
    }

    annotations_by_filename: dict[
        str,
        ImageAnnotation,
    ] = {}

    for annotation in annotations:
        if (
            annotation.filename
            in annotations_by_filename
        ):
            raise ValueError(
                "Duplicate annotation for "
                f"{annotation.filename}"
            )

        annotations_by_filename[
            annotation.filename
        ] = annotation

    _export_split(
        split_name="train",
        filenames=split.train,
        source_images_directory=(
            source_images_directory
        ),
        annotations_by_filename=(
            annotations_by_filename
        ),
        output_directory=output_directory,
        class_to_index=class_to_index,
    )

    _export_split(
        split_name="val",
        filenames=split.validation,
        source_images_directory=(
            source_images_directory
        ),
        annotations_by_filename=(
            annotations_by_filename
        ),
        output_directory=output_directory,
        class_to_index=class_to_index,
    )

    _export_split(
        split_name="test",
        filenames=split.test,
        source_images_directory=(
            source_images_directory
        ),
        annotations_by_filename=(
            annotations_by_filename
        ),
        output_directory=output_directory,
        class_to_index=class_to_index,
    )

    return _write_dataset_yaml(
        output_directory,
        class_names,
    )


def _instance_to_yolo_line(
        instance: InstanceAnnotation,
        class_to_index: dict[str,int],
) -> str:
    if instance.class_name not in class_to_index:
        raise ValueError(
            f"Unknown class: {instance.class_name}"
        )

    class_index = class_to_index[
        instance.class_name
    ]

    values = [
        str(class_index)
    ]

    for point in instance.polygon:
        values.append(
            f"{point.x:.6f}"
        )
        values.append(
            f"{point.y:.6f}"
        )

    return " ".join(values)

def _write_annotation(
        annotation: ImageAnnotation,
        output_path: Path,
        class_to_index: dict[str, int],
) -> None:
    lines = [
        _instance_to_yolo_line(
            instance,
            class_to_index,
        )
        for instance in annotation.instances
    ]

    content = ""

    if lines:
        content = (
            "\n".join(lines)
            + "\n"
        )

    output_path.write_text(
        content,
        encoding="utf-8",
    )

def _export_split(
        split_name: str,
        filenames: Sequence[str],
        source_images_directory: Path,
        annotations_by_filename: dict[
            str,
            ImageAnnotation,
        ],
        output_directory: Path,
        class_to_index: dict[str, int],
) -> None:
    images_directory = (
        output_directory
        / "images"
        / split_name
    )

    labels_directory = (
        output_directory
        / "labels"
        / split_name
    )

    images_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    labels_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    for filename in filenames:
        source_image = (
            source_images_directory
            / filename
        )

        if not source_image.exists():
            raise FileNotFoundError(
                f"Dataset image does not exist: "
                f"{source_image}"
            )

        annotation = (
            annotations_by_filename.get(
                filename
            )
        )

        if annotation is None:
            raise ValueError(
                f"No annotation exists for "
                f"{filename}"
            )

        destination_image = (
            images_directory
            / filename
        )

        label_path = (
            labels_directory
            / f"{Path(filename).stem}.txt"
        )

        _write_annotation(
            annotation,
            label_path,
            class_to_index,
        )

def _write_dataset_yaml(
        output_directory: Path,
        class_names: Sequence[str],
) -> Path:
    yaml_path = (
        output_directory
        / "dataset.yaml"
    )

    payload = {
        "path": str(
            output_directory.resolve()
        ),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {
            index: name
            for index, name
            in enumerate(class_names)
        },
    }

    yaml_path.write_text(
        yaml.safe_dump(
            payload,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    return yaml_path