import numpy as np
import pytest

from pathlib import Path

import cv2

from tabletop_vision.calibration import(
    CameraCalibrationResult,
    CalibrationObservation,
    CharucoBoardSpec,
    calibrate_camera,
    create_charuco_board,
    create_charuco_board_image,
    millimetres_to_pixels,
    create_undistortion_maps,
)

def test_default_board_dimensios() -> None:
    spec = CharucoBoardSpec()

    assert spec.board_width_mm == pytest.approx(175.0)
    assert spec.board_height_mm == pytest.approx(125.0)
    assert spec.internal_corner_count == 24

def test_millimetres_to_pixels() -> None:
    pixels = millimetres_to_pixels(
        length_mm=25.4,
        dpi=300
    )

    assert pixels == 300

def test_marker_must_be_smaller_than_square() -> None:
    with pytest.raises(ValueError):
        CharucoBoardSpec(
            square_length_mm=25.0,
            marker_length_mm=25.0,
        )

    def test_board_image_is_generated() -> None:
        spec = CharucoBoardSpec()

        image = create_charuco_board_image(
            spec,
            dpi=100,
        )

        expected_width = millimetres_to_pixels(
            spec.board_width_mm,
            100,
        )

        expected_height = millimetres_to_pixels(
            spec.board_height_mm,
            100,
        )

        assert image.shape == (
            expected_height,
            expected_width
        )

        assert image.dtype == np.uint8
        assert image.min() == 0
        assert image.max() == 255

def test_calibration_recovers_synthetic_camera() -> None:
    spec = CharucoBoardSpec()
    board = create_charuco_board(spec)

    object_points = np.asarray(
        board.getChessboardCorners(),
        dtype=np.float32,
    ).reshape(-1,3)

    expected_camera_matrix = np.array(
        [
            [900.0, 0.0, 640.0],
            [0.0, 910.0, 360.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    expected_distortion = np.array(
        [
            0.08,
            -0.03,
            0.001,
            0.0
        ],
        dtype=np.float64
    )

    observations: list[CalibrationObservation] = []


    # Generate synthetic observations by projecting the known board points into the image using a synthetic camera model.
    # We vary the rotation and translation of the board to simulate different views.

    for index in range(15):
        rotation_vector = np.array(
            [
                0.05 + 0.02 * index,
                -0.15 + 0.02 * (index % 5),
                -0.20 + 0.03 * index,
             ],
             dtype = np.float64,
        )

        translation_vector = np.array(
            [
                -0.08 + 0.012 * index,
                -0.05 + 0.01 * (index % 4),
                0.45 + 0.025 * (index % 3),
            ],
            dtype=np.float64,
        )

        image_points, _ = cv2.projectPoints(
            object_points,
            rotation_vector,
            translation_vector,
            expected_camera_matrix,
            expected_distortion,
        )

        image_points_asarray = np.asarray(image_points,dtype=np.float32)

        observations.append(
            CalibrationObservation(
                image_path=Path(f"synthetic_{index:03d}.png"),
                object_points=object_points.copy(),
                image_points=image_points_asarray,
            )
        )

    result = calibrate_camera(
        observations=observations,
        image_size=(1280, 720),
    )

    assert (
        result.rms_reprojection_error
        < 0.001
    )

    assert np.allclose(
        result.camera_matrix,
        expected_camera_matrix,
        atol=0.1,
    )

    assert len(result.rotation_vectors) == 15
    assert len(result.translation_vectors) == 15
    assert len(result.per_view_errors) == 15


def test_undistortion_maps_match_image_size() -> None:
    calibration = CameraCalibrationResult(
        image_size=(1280, 720),
        rms_reprojection_error=0.0,
        camera_matrix=np.array(
            [
                [900.0, 0.0, 640.0],
                [0.0, 900.0, 360.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64
        ),
        distortion_coefficients=np.array(
            [0.1, -0.05, 0.001, -0.001, 0.0],
            dtype=np.float64
        ),
        rotation_vectors=(),
        translation_vectors=(),
        per_view_errors=(),
    )

    maps = create_undistortion_maps(calibration)

    assert maps.map_x.shape == (720, 1280)
    assert maps.map_y.shape == (720, 1280)
    
        