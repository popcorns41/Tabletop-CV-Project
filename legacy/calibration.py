from __future__ import annotations

from dataclasses import dataclass

import json
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np


"""Depreciated version of calibration, started getting too long so newer version decomposes
functionality across a calibration package"""

MILLIMETRES_PER_INCH = 25.4

# Default board dimensions are:
# 7 squares wide
# 5 squares tall
# 25 mm per square
# 18 mm per ArUco market

#Physical dimensions:
#Width: 7 x 25 mm | 175 mm^2
#Height: 5 x 25 mm | 125 mm^2

@dataclass(frozen=True, slots=True)
class CharucoBoardSpec:
    """Physical and visual specification of a ChArUco Board."""

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

        if not hasattr(cv2.aruco, self.dictionary_name):
            raise ValueError(
                f"Unknown ArUco dictionary: {self.dictionary_name}"
            )

    @property
    def board_width_mm(self) -> float:
        return self.squares_x * self.square_length_mm

    @property
    def board_height_mm(self) -> float:
        return self.squares_y * self.square_length_mm

    @property
    def internal_corner_count(self) -> int:
        return (self.squares_x - 1) * (self.squares_y -1)

    @property
    def square_length_metres(self) -> float:
        return self.square_length_mm / 1000.0

    @property
    def marker_length_metres(self) -> float:
        return self.marker_length_mm / 1000.0

@dataclass(frozen=True, slots=True)
class CharucoDetection:
    """Marker and ChArUco corner observations from one image."""

    charuco_corners: np.ndarray | None
    charuco_ids: np.ndarray | None
    marker_corners: Sequence[np.ndarray]
    marker_ids: np.ndarray | None

    @property
    def corner_count(self) -> int:
        if self.charuco_ids is None:
            return 0

        return len(self.charuco_ids)

    @property
    def marker_count(self) -> int:
        if (self.marker_ids is None):
            return 0

        return len(self.marker_ids)

@dataclass(frozen=True, slots=True)
class ClaibrationObservation:
    """Known board points and their detected image positions in one view."""
    image_path: Path
    object_points: np.ndarray
    image_points: np.ndarray

    @property
    def point_count(self) -> int:
        return len(self.object_points)


def millimetres_to_pixels(
        length_mm: float,
        dpi: int,
) -> int:
    """Convert a physical length into pixel at a given DPI."""

    if length_mm <= 0:
        raise ValueError("length_mm must be positive.")

    if dpi <= 0:
        raise ValueError("dpi must be positive.")

    inches = length_mm / MILLIMETRES_PER_INCH
    return round(inches * dpi)

def create_charuco_board(
    spec: CharucoBoardSpec,
) -> cv2.aruco.CharucoBoard:
    """Create the OpenCV ChArUco board described by the specification."""

    dictionary_id = getattr(
        cv2.aruco,
        spec.dictionary_name,
    )

    dictionary = cv2.aruco.getPredefinedDictionary(
        dictionary_id
    )

    return cv2.aruco.CharucoBoard(
        (spec.squares_x, spec.squares_y),
        spec.square_length_metres,
        spec.marker_length_metres,
        dictionary,
    )

def create_charuco_board_image(
        spec: CharucoBoardSpec,
        dpi: int = 300,
) -> np.ndarray:
    """Generate a printable grayscale ChArUco board image."""

    board = create_charuco_board(spec)

    image_width = millimetres_to_pixels(
        spec.board_width_mm,
        dpi,
    )

    image_height = millimetres_to_pixels(
        spec.board_height_mm,
        dpi,
    )

    return board.generateImage(
        (image_width, image_height),
        marginSize=0,
        borderBits = spec.border_bits,
    )