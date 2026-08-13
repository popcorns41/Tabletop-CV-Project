from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from tabletop_vision.dataset.models import (
    DatasetImageMetadata,
    ImageAnnotation,
)

from tabletop_vision.evaluation.benchmark import (
    BenchmarkSample,
)

from tabletop_vision.evaluation.models import (
    GroundTruthPose
)

from tabletop_vision.perception.contours import (
    contour_centroid,
    contour_rotated_rectangle,
    polygon_to_contour,
)



def annotation_to_ground_truth_pose(
        annotation: ImageAnnotation,
        metadata: DatasetImageMetadata,
        class_name: str,
) -> GroundTruthPose | None:
    target_instances = [
        instance
        for instance in annotation.instances
        if instance.class_name == class_name
    ]

    if not target_instances:
        return None

    if len(target_instances) > 1:
        raise ValueError(
            f"Expected at most one '{class_name}' "
            f"instance in {annotation.filename}, "
            f"found {len(target_instances)}."
        )

    instance = target_instances[0]

    #We can move from dataset coordinate space [0, 1]
    # to image pixel coordinate space by multipling x and y
    #by width and height in px 

    polygon = np.array(
        [
            [
                point.x * metadata.width,
                point.y * metadata.height,
            ]
            for point in instance.polygon
        ],
        dtype=np.float32,
    )

    contour = polygon_to_contour(
        polygon
    )

    centroid = contour_centroid(contour)

    if centroid is None:
        raise ValueError(
            f"Could not calculate centroid "
            f"for {annotation.filename}."
        )

    rotated_rectangle = (
        contour_rotated_rectangle(
            contour
        )
    )

    return GroundTruthPose(
        centroid=(
            float(centroid[0]),
            float(centroid[1]),
        ),
        angle_degrees=float(
            rotated_rectangle.angle_degrees
        ),
    )


def create_benchmark_sample(
        filename: str,
        images_directory: Path,
        metadata: DatasetImageMetadata,
        annotation: ImageAnnotation,
        class_name: str,
) -> BenchmarkSample:
    if metadata.filename != filename:
        raise ValueError(
            f"Metadata filename '{metadata.filename}' "
            f"does not match '{filename}'."
        )

    if annotation.filename != filename:
        raise ValueError(
            f"Annotation filename '{annotation.filename}' "
            f"does not match '{filename}'."
        )

    image_path = (
        images_directory
        / filename
    )

    frame = cv2.imread(
        str(image_path)
    )

    if frame is None:
        raise ValueError(
            f"Could not load image: "
            f"{image_path}"
        )

    actual_height, actual_width = (
        frame.shape[:2]
    )

    if (
        actual_width != metadata.width
        or actual_height != metadata.height
    ):
        raise ValueError(
            f"Image dimensions for {filename} "
            f"do not match metadata."
        )
    
    ground_truth = (
        annotation_to_ground_truth_pose(
            annotation=annotation,
            metadata=metadata,
            class_name=class_name,
        )
    )

    return BenchmarkSample(
        frame=frame,
        ground_truth= ground_truth,
    )

def create_benchmark_samples(
        filenames: Sequence[str],
        images_directory: Path,
        metadata: Sequence[DatasetImageMetadata],
        annotations: Sequence[ImageAnnotation],
        class_name: str,
) -> tuple[BenchmarkSample, ...]:
    metadata_by_filename = {
        item.filename: item
        for item in metadata
    }

    annotations_by_filename = {
        item.filename: item
        for item in annotations
    }

    if len(metadata_by_filename) != len(metadata):
        raise ValueError(
            "Duplicate metadata filenames "
            "were provided."
        )

    if len(annotations_by_filename) != len(annotations):
        raise ValueError(
            "Duplicate annotation filenames "
            "were provided."
        )

    samples: list[BenchmarkSample] = []

    for filename in filenames:
        metadata_item = (
            metadata_by_filename.get(
                filename
            )
        )

        if metadata_item is None:
            raise ValueError(
                f"No metdata found for "
                f"{filename}."
            )

        annotation = (
            annotations_by_filename.get(
                filename
            )
        )

        if annotation is None:
            raise ValueError(
                f"No annotation found for "
                f"{filename}."
            )

        sample = create_benchmark_sample(
            filename=filename,
            images_directory=images_directory,
            metadata=metadata_item,
            annotation=annotation,
            class_name=class_name
        )

        samples.append(
            sample
        )

    return tuple(samples)
