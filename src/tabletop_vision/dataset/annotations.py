from __future__ import annotations

import json
from pathlib import Path

from tabletop_vision.dataset.models import (
    ImageAnnotation,
    InstanceAnnotation,
    PolygonPoint,
)

class AnnotationWriter:
    """Persist image-level segmentation annotations as JSON Lines."""

    def __init__(
            self,
            output_path: Path,
    ) -> None:
        self.__output_path = output_path

        self.__output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    def append(
            self,
            annotation: ImageAnnotation,
    ) -> None:
        payload = {
            "filename": annotation.filename,
            "instances": [
                {
                    "class_name": instance.class_name,
                    "polygon": [
                        [
                            point.x,
                            point.y,
                        ]
                        for point in instance.polygon
                    ],
                }
                for instance in annotation.instances
            ],
        }

        with self.__output_path.open(
            "a",
            encoding="utf-8",
        ) as file:
            file.write(
                json.dumps(payload)
                + "\n"
            )

def load_annotations(
        input_path: Path,
) -> list[ImageAnnotation]:
    """Load image annotations from a JSON Lines file."""

    if not input_path.exists():
        raise FileNotFoundError(
            f"Annotation file does not exist: {input_path}"
        )

    annotations: list[ImageAnnotation] = []

    with input_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            stripped = line.strip()

            if not stripped:
                continue

            try:
                payload = json.loads(stripped)

            except json.JSONDecodeError as error:
                raise ValueError(
                    "Invalid annotation JSON on "
                    f"line {line_number}."
                ) from error

            instances = tuple(
                InstanceAnnotation(
                    class_name=instance[
                        "class_name"
                    ],
                    polygon=tuple (
                        PolygonPoint(
                            x=float(point[0]),
                            y=float(point[1]),
                        )
                        for point in instance[
                            "polygon"
                        ]
                    ),
                )
                for instance in payload[
                    "instances"
                ]
            )

            annotations.append(
                ImageAnnotation(
                    filename=payload[
                        "filename"
                    ],
                    instances=instances,
                )
            )
    return annotations
        