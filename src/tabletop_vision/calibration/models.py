from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True, slots=True)
class CharucoBoardSpec:
    squares_x: int = 7
    squares_y: int = 5

    square_length_mm: float = 25.0
    marker_length_mm: float = 18.0

    dictionary_name: str = "DICT_5X5_100"
    border_bits: int = 1

    def __post_init__(self) -> None:
        if self.squares_x < 2:
            raise ValueError("squares_x must be at least 2.")

        if self.squares_y < 2:
            raise ValueError("squares_y must be at least 2.")

        if self.square_length_mm <= 0:
            raise ValueError("square_length_mm must be positive.")

        if self.marker_length_mm <= 0:
            raise ValueError("marker_length_mm must be positive.")

        if self.marker_length_mm >= self.square_length_mm:
            raise ValueError(
                "marker_length_mm must be smaller than square_length_mm."
            )

        if self.border_bits < 1:
            raise ValueError("border_bits must be at least 1.")

    @property
    def board_width_mm(self) -> float:
        return self.squares_x * self.square_length_mm

    @property
    def board_height_mm(self) -> float:
        return self.squares_y * self.square_length_mm

    @property
    def internal_corner_count(self) -> int:
        return (self.squares_x - 1) * (self.squares_y - 1)

    @property
    def square_length_metres(self) -> float:
        return self.square_length_mm / 1000.0

    @property
    def marker_length_metres(self) -> float:
        return self.marker_length_mm / 1000.0


@dataclass(frozen=True, slots=True)
class CharucoDetection:
    charuco_corners: np.ndarray | None
    charuco_ids: np.ndarray | None
    marker_corners: Sequence[np.ndarray]
    marker_ids: np.ndarray | None

    @property
    def corner_count(self) -> int:
        return (
            0
            if self.charuco_ids is None
            else len(self.charuco_ids)
        )

    @property
    def marker_count(self) -> int:
        return (
            0
            if self.marker_ids is None
            else len(self.marker_ids)
        )


@dataclass(frozen=True, slots=True)
class CalibrationObservation:
    image_path: Path
    object_points: np.ndarray
    image_points: np.ndarray

    @property
    def point_count(self) -> int:
        return len(self.object_points)


@dataclass(frozen=True, slots=True)
class CameraCalibrationResult:
    image_size: tuple[int, int]

    rms_reprojection_error: float
    camera_matrix: np.ndarray
    distortion_coefficients: np.ndarray

    rotation_vectors: Sequence[np.ndarray]
    translation_vectors: Sequence[np.ndarray]

    per_view_errors: tuple[float, ...]

@dataclass(frozen=True, slots=True)
class UndistortionMaps:
    """Precomputed pixel mappings for real-time image undistortion."""

    map_x: np.ndarray
    map_y: np.ndarray
    camera_matrix: np.ndarray