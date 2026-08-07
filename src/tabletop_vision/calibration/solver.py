import numpy as np
import cv2
from typing import Sequence

from tabletop_vision.calibration.models import (
    CalibrationObservation,
    CameraCalibrationResult
)

def calculate_reprojection_rmse(
        observation: CalibrationObservation,
        rotation_vector: np.ndarray,
        translation_vector: np.ndarray,
        camera_matrix: np.ndarray,
        distortion_coefficients: np.ndarray,
) -> float:
    """Calculate pixel RMSE for one calibration view."""

    projected_points, _ = cv2.projectPoints(
        observation.object_points,
        rotation_vector,
        translation_vector,
        camera_matrix,
        distortion_coefficients
    )

    observed = observation.image_points.reshape(-1,2)

    projected = np.asarray(
        projected_points,
        dtype=np.float64,
    ).reshape(-1,2)

    residuals = observed - projected

    squared_pixel_errors = np.sum(
        residuals**2,
        axis=1,
    )

    return float(
        np.sqrt(
            np.mean(squared_pixel_errors)
        )
    )

def calibrate_camera(
        observations: Sequence[CalibrationObservation],
        image_size: tuple[int, int],
) -> CameraCalibrationResult:
    """Estimat ecamera intrinsics, distortion and board poses."""

    if len(observations) < 3:
        raise ValueError(
            "At least three calibration observations are required."
        )

    image_width, image_height = image_size

    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive.")

    object_points = [
        observation.object_points
        for observation in observations
    ]

    image_points = [
        observation.image_points
        for observation in observations
    ]

    (rms_error,
     camera_matrix,
     distortion_coefficients,
     rotation_vectors,
     translation_vectors
     ) = cv2.calibrateCamera(
         object_points,
         image_points,
         image_size,
         None,
         None,
     )

    # Bananas tuple unpacking to avoid "too many values to unpack" error

    per_view_errors = tuple(
        calculate_reprojection_rmse(
            observation=observation,
            rotation_vector=rotation_vector,
            translation_vector=translation_vector,
            camera_matrix=camera_matrix,
            distortion_coefficients=distortion_coefficients,
        )
        for (observation, rotation_vector, translation_vector)
        in zip (observations, rotation_vectors, translation_vectors, strict=True,)
    ) 

    camera_matrix_asarray = np.asarray(camera_matrix,dtype=np.float64)
    distortion_coefficients_asarray = np.asarray(distortion_coefficients,dtype=np.float64)
    return CameraCalibrationResult(
        image_size=image_size,
        rms_reprojection_error=float(rms_error),
        camera_matrix=camera_matrix_asarray,
        distortion_coefficients=distortion_coefficients_asarray,
        rotation_vectors=rotation_vectors,
        translation_vectors=translation_vectors,
        per_view_errors=per_view_errors,
    )