from __future__ import annotations

from dataclasses import dataclass
from typing import Self

import cv2
import numpy as np

class CameraError(RuntimeError):
    """Raised when the webcam cannot be opened or read."""

@dataclass(frozen=True, slots=True)
class CameraConfig:
    """Requested webcam capture settings."""

    index: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30

@dataclass(frozen=True, slots=True)
class CameraProperties:
    """Capture settings reproted by the active webcam backend."""

    width: int
    height: int
    fps: float
    backend: str

class Webcam:
    """Owns an OpenCV videoCapture resource."""

    def __init__(self, config: CameraConfig) -> None:
        self.config = config
        # _capture is initialized to None to avoid 
        # creating a VideoCapture object in the constructor.
        self._capture: cv2.VideoCapture | None = None

    def open(self) -> None:
        if self.is_open:
            return

        #macOS camera backend supported
        capture = cv2.VideoCapture(self.config.index, cv2.CAP_AVFOUNDATION)

        if not capture.isOpened():
            capture.release()
            raise CameraError(
                f"Could not open camera index {self.config.index}. "
                "Try another index such as --camera 1."
            )

        # These are requests. The camera or backend may choose different values.
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        capture.set(cv2.CAP_PROP_FPS, self.config.fps)

        self._capture = capture

    @property
    def is_open(self) ->bool:
        return self._capture is not None and self._capture.isOpened()

    def properties(self) -> CameraProperties:
        capture = self._require_capture()

        try:
            backend = capture.getBackendName()
        except cv2.error:
            backend = "unknown"

        return CameraProperties(
            width = round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height = round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps = capture.get(cv2.CAP_PROP_FPS),
            backend=backend
        )

    def read(self) -> np.ndarray:
        capture = self._require_capture()

        successful, frame = capture.read()

        if not successful or frame is None or frame.size == 0:
            raise CameraError("The webcam failed to provide a valid frame.")

        return frame

    def release(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def _require_capture(self) -> cv2.VideoCapture:
        if not self.is_open or self._capture is None:
            raise CameraError("The webcam is not opne.")

        return self._capture

    def __enter__(self) -> Self:
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.release()