from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from tabletop_vision.dataset.models import (
    DataImageMetadata,
)

class DatasetWriter:
    """Persist captured images and their associated metadata."""

    def __init__(
            self,
            root_directory: Path,
            environment: str | None = None,
    ) -> None:
        self._root_directory = root_directory
        self._images_directory = (
            root_directory / "images"
        )

        self._metadata_path = (
            root_directory / "metadata.jsonl"
        )

        self._environment = environment

        self._images_directory.mkdir(
            parents=True,
            exist_ok=True
        )

        self._next_index = self._find_next_index()

    @property
    def image_count(self) -> int:
        return self._next_index - 1

    def save_frame(
            self,
            frame: np.ndarray,
    ) -> DataImageMetadata:
        height, width = frame.shape[:2]

        filename = (
            f"frame_{self._next_index:0.06d}.jpg"
        )

        image_path = (
            self._images_directory / filename
        )

        success = cv2.imwrite(
            str(image_path),
            frame,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                95,
            ],
        )

        if not success:
            raise RuntimeError(
                f"Failed to save image: {image_path}"
            )

        metadata = DataImageMetadata(
            filename=filename,
            width=width,
            height=height,
            timestamp=(
                datetime.now(timezone.utc)
                .isoformat()
            ),
            environments=self._environment,
        )

        self._append_metadata(
            metadata
        )

        self._next_index += 1

        return metadata

    def _find_next_index(self) -> int:
        existing_indices: list[int] = []

        for path in self._images_directory.glob(
            "frame_*.jpg"
        ):
            try:
                index = int(
                    path.stem.removeprefix(
                        "frame_"
                    )
                )
            except ValueError:
                continue

            existing_indices.append(index)

        if not existing_indices:
            return 1

        return max(existing_indices) + 1

    def _append_metadata(
            self,
            metadata: DataImageMetadata,
    ) -> None:
        payload = {
            "filename" : metadata.filename,
            "width" : metadata.width,
            "height" : metadata.height,
            "timestamp" : metadata.timestamp,
            "environment" : metadata.environments,
        }

        with self._metadata_path.open(
            "a",
            encoding="utf-8",
        ) as file:
            file.write(
                json.dumps(payload)
                + "\n"
            )