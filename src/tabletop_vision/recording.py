from __future__ import annotations

from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

class RecordingError(RuntimeError):
    """Raised when a snapshot or recording cannot be written."""

def create_timestamp() -> str:
    """Return a filename-safe timestamp with millisecond precision."""

    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

# save_snapshot saves a single frame to disk as a PNG image. The filename is based on the current timestamp.
def save_snapshot(frame: np.ndarray, output_directory: Path) -> Path:
    output_directory.mkdir(parents = True, exist_ok=True)

    output_path = output_directory / f"capture_{create_timestamp()}.png"
    successful = cv2.imwrite(str(output_path), frame)

    if not successful:
        raise RecordingError(f"Could not save snapshot to {output_path}.")

    return output_path

class VideoRecorder:
    """Writes raw webcam frames to a reusable AVI test recording."""

    def __init__(self, output_directory: Path) -> None:
        self.output_directory = output_directory
        self._writer: cv2.VideoWriter | None = None
        self._path: Path | None = None
        self._frame_size: tuple[int, int] | None = None

    @property
    def is_recording(self) -> bool:
        return self._writer is not None

    @property
    def current_path(self) -> Path | None:
        return self._path

    def start(self, frame: np.ndarray, fps: float) -> Path:
        if self.is_recording:
            raise RecordingError("A recording is already active.")

        self.output_directory.mkdir(parents=True, exist_ok=True)

        height, width = frame.shape[:2]
        frame_size = (width,height)
        output_path = self.output_directory / f"recording_{create_timestamp()}.avi"

        # MJPG in an AVI container is a pragmatic format for resuable CV footage.
        fourcc = cv2.VideoWriter.fourcc(*"MJPG")

        writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            fps,
            frame_size,
        )

        if not writer.isOpened():
            writer.release()
            raise RecordingError(f"Could not create recording at {output_path}.")

        self._writer = writer
        self._path = output_path
        self._frame_size = frame_size

        return output_path

    def write(self, frame: np.ndarray) -> None:
        if self._writer is None or self._frame_size is None:
            raise RecordingError("No recording is active.")

        height, width = frame.shape[:2]
        current_size = (width, height)

        if current_size != self._frame_size:
            raise RecordingError(
                f"Frame size changed from {self._frame_size} to {current_size}"
                "during recording."
            )

        self._writer.write(frame)

    def stop(self) -> Path | None:
        if self._writer is None:
            return None

        self._writer.release()

        completed_path = self._path

        self._writer = None
        self._path = None
        self._frame_size = None

        return completed_path
        