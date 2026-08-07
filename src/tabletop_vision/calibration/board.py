from __future__ import annotations

import cv2
import numpy as np

from tabletop_vision.calibration.models import CharucoBoardSpec


MILLIMETRES_PER_INCH = 25.4


def millimetres_to_pixels(
    length_mm: float,
    dpi: int,
) -> int:
    if length_mm <= 0:
        raise ValueError("length_mm must be positive.")

    if dpi <= 0:
        raise ValueError("dpi must be positive.")

    inches = length_mm / MILLIMETRES_PER_INCH

    return round(inches * dpi)


def create_charuco_board(
    spec: CharucoBoardSpec,
) -> cv2.aruco.CharucoBoard:
    if not hasattr(
        cv2.aruco,
        spec.dictionary_name,
    ):
        raise ValueError(
            f"Unknown ArUco dictionary: "
            f"{spec.dictionary_name}"
        )

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
    board = create_charuco_board(spec)

    width = millimetres_to_pixels(
        spec.board_width_mm,
        dpi,
    )

    height = millimetres_to_pixels(
        spec.board_height_mm,
        dpi,
    )

    return board.generateImage(
        (width, height),
        marginSize=0,
        borderBits=spec.border_bits,
    )