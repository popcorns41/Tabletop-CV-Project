from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from tabletop_vision.dataset.models import (
    DatasetImageMetadata,
    DatasetSplit,
    DatasetValidationReport,
    ImageAnnotation,
)

def validate_dataset(
        metadata: Sequence[DatasetImageMetadata],
        annotations: Sequence[ImageAnnotation],
        split: DatasetSplit,
        images_directory: Path,
        allowed_classes: set[str],
) -> DatasetValidationReport:
    """Validate structural consistency of a dataset."""

    errors: list[str] = []
    warnings: list[str] = []

    _validate_metadata(
        metadata=metadata,
        images_directory=images_directory,
        errors=errors,
    )

    _validate_orphan_images(
        metadata=metadata,
        images_directory=images_directory,
        warnings=warnings
    )

    _validate_annotations(
        metadata=metadata,
        annotations=annotations,
        allowed_classes=allowed_classes,
        errors=errors,
        warnings=warnings,
    )

    _validate_split(
        split=split,
        metadata=metadata,
        errors=errors,
        warnings=warnings,
    )

    return DatasetValidationReport(
        errors=tuple(errors),
        warnings=tuple(warnings),
    )

def _find_duplicates(
        values: Sequence[str],
) -> set[str]:
    counts = Counter(values)

    return {
        value
        for value, count in counts.items()
        if count > 1
    }


#Catch inconsistencies between metadata and actual images in persistence
def _validate_metadata(
        metadata: Sequence[DatasetImageMetadata],
        images_directory: Path,
        errors: list[str],
) -> None:
    filenames = [
        item.filename
        for item in metadata
    ]

    duplicates = _find_duplicates(filenames)

    for filename in sorted(duplicates):
        errors.append(
            f"Duplicate metadata entry: {filename}"
        )

    for item in metadata:
        image_path = (
            images_directory / item.filename
        )

        if not image_path.exists():
            errors.append(
                f"Metadata references missing image: "
                f"{item.filename}"
            )

        if item.width <= 0 or item.height <= 0:
            errors.append(
                f"Invalid image dimensions for "
                f"{item.filename}: "
                f"{item.width}x{item.height}"
            )

def _validate_orphan_images(
        metadata: Sequence[DatasetImageMetadata],
        images_directory: Path,
        warnings: list[str],
) -> None:
    known_images = {
        item.filename
        for item in metadata
    }

    image_files = {
        path.name
        for path in images_directory.iterdir()
        if (
            path.is_file()
            and path.suffix.lower()
            in  {
                ".jpg",
                ".jpeg",
                ".png",
            }
        )
    }

    orphaned = (
        image_files - known_images
    )

    for filename in sorted(orphaned):
        warnings.append(
            f"Image has no metadata: {filename}"
        )

def _validate_annotations(
    metadata: Sequence[DatasetImageMetadata],
    annotations: Sequence[ImageAnnotation],
    allowed_classes: set[str],
    errors: list[str],
    warnings: list[str],
) -> None:
    metadata_by_filename = {
        item.filename: item
        for item in metadata
    }

    annotation_filenames = [
        annotation.filename
        for annotation in annotations
    ]

    duplicates = _find_duplicates(
        annotation_filenames
    )

    for filename in sorted(duplicates):
        errors.append(
            f"Duplicate annotation entry: {filename}"
        )

    for annotation in annotations:
        if (
            annotation.filename
            not in metadata_by_filename
        ):
            errors.append(
                "Annotation references unknown image: "
                f"{annotation.filename}"
            )

            continue

        metadata_item = metadata_by_filename[
            annotation.filename
        ]

        for instance in annotation.instances:
            if (
                instance.class_name
                not in allowed_classes
            ):
                errors.append(
                    f"Unknown class "
                    f"'{instance.class_name}' in "
                    f"{annotation.filename}"
                )

        if (
            metadata_item.target_present
            and not annotation.instances
        ):
            warnings.append(
                f"{annotation.filename} is marked "
                "target_present=True but contains "
                "no annotated instances."
            )

        if (
            not metadata_item.target_present
            and annotation.instances
        ):
            warnings.append(
                f"{annotation.filename} is marked "
                "target_present=False but contains "
                "annotated instances."
            )

    annotated_images = set(
        annotation_filenames
    )

    for item in metadata:
        if item.filename not in annotated_images:
            warnings.append(
                f"Image has not been annotated: "
                f"{item.filename}"
            )
            
def _validate_split(
        split: DatasetSplit,
        metadata: Sequence[DatasetImageMetadata],
        errors: list[str],
        warnings: list[str],
) -> None:
    train = set(split.train)
    validation = set(split.validation)
    test = set(split.test)

    train_validation_overlap = (
        train & validation
    )

    train_test_overlap = (
        train & test
    )

    validation_test_overlap =(
        validation & test
    )

    for filename in sorted(
        train_validation_overlap
    ):
        errors.append(
            f"Split leakage: {filename} appears "
            "in train and validation."
        )

    for filename in sorted(
        train_test_overlap
    ):
        errors.append(
            f"Split leakage: {filename} appears "
            "in train and test."
        )

    for filename in sorted(
        validation_test_overlap
    ):
        errors.append(
            f"Split leakage: {filename} appears "
            "in validation and test."
        )

    known_images = {
        item.filename
        for item in metadata
    }

    assigned_images = (
        train
        | validation
        | test
    )

    unknown = (
        assigned_images - known_images
    )

    for filename in sorted(unknown):
        errors.append(
            f"Split references unknown image: "
            f"{filename}"
        )

    unassigned = (
        known_images - assigned_images
    )

    for filename in sorted(unassigned):
        warnings.append(
            f"Image is not assigned to a split: "
            f"{filename}"
        )