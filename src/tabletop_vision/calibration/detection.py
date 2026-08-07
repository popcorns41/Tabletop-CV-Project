from pathlib import Path

import cv2
import numpy as np

from tabletop_vision.calibration.board import (
    create_charuco_board,
)
from tabletop_vision.calibration.models import (
    CalibrationObservation,
    CharucoBoardSpec,
    CharucoDetection,
)

def create_charuco_detector(
        spec: CharucoBoardSpec,
) -> cv2.aruco.CharucoDetector:
    """Create a detector configured from the specified board."""

    board = create_charuco_board(spec)

    return cv2.aruco.CharucoDetector(board)

def detect_charuco_board(
        detector: cv2.aruco.CharucoDetector,
        frame: np.ndarray
) -> CharucoDetection:
    """Detect ChArUco markers and interpolated corners in one frame."""

    grayscale = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY,
    )

    (charuco_corners,charuco_ids,marker_corners,marker_ids) = detector.detectBoard(grayscale)

    return CharucoDetection(
        charuco_corners=charuco_corners,
        charuco_ids=charuco_ids,
        marker_corners=marker_corners,
        marker_ids=marker_ids,
    )


def create_calibration_observation(
        board: cv2.aruco.CharucoBoard,
        detection: CharucoDetection,
        image_path: Path,
) -> CalibrationObservation:
    """Match detected corners IDs to know physical board coordinates."""

    if (
        detection.charuco_corners is None
        or detection.charuco_ids is None
    ):
        raise ValueError(
            "The detection does not contain any ChArUco corners."
        )

    # corner_ids is a 1D array of the IDs of the detected corners, 
    # in the same order as charuco_corners
    corner_ids = np.asarray(
        detection.charuco_ids,
        dtype = np.int32,
    ).reshape(-1)

    # image_points is a 3D array of the detected corner positions in the image,
    # in the same order as corner_ids
    image_points = np.asarray(
        detection.charuco_corners,
        dtype=np.float32,
    ).reshape(-1,1,2)

    # all_board_points is a 3D array of the known physical positions of all corners on the board,
    # in the same order as the corner IDs
    all_board_points = np.asarray(
        board.getChessboardCorners(),
        dtype=np.float32,
    ).reshape(-1,3)

    if np.any(corner_ids < 0):
        raise ValueError("A detected ChArUco ID was negative.")

    if np.any(corner_ids >= len(all_board_points)):
        raise ValueError(
            "A detected ChArUco ID exceeds the board corner count."
        )

    object_points = all_board_points[corner_ids]

    if len(object_points) != len(image_points):
        raise ValueError(
            "Object-point and image-point counts do not match."
        )

    return CalibrationObservation(
        image_path=image_path,
        object_points=object_points,
        image_points=image_points
    )


