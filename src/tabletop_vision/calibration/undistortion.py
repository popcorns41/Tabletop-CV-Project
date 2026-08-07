import numpy as np
import cv2

from tabletop_vision.calibration.models import (
    UndistortionMaps,
    CameraCalibrationResult
)

#For every output pixel, we determine which location in the
#distorted original image corresponds to it.

def undistort_frame(
        frame: np.ndarray,
        camera_matrix: np.ndarray,
        distortion_coefficients: np.ndarray,
) -> np.ndarray:
    """Remove lens distortion form a camera frame."""

    return cv2.undistort(
        frame,
        camera_matrix,
        distortion_coefficients
    )


#Precompute expensive geometric map before runtime
def create_undistortion_maps(
        calibration: CameraCalibrationResult,
) -> UndistortionMaps:
    """Precompute mappings used to undisort camera frames."""

    width, height = calibration.image_size

    new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
        calibration.camera_matrix,
        calibration.distortion_coefficients,
        (width,height),
        1.0,
        (width, height),
    )

    map_x, map_y = cv2.initUndistortRectifyMap(
        calibration.camera_matrix,
        calibration.distortion_coefficients,
        None,
        new_camera_matrix,
        (width, height),
        cv2.CV_32FC1,
    )

    return UndistortionMaps(
        map_x=map_x,
        map_y=map_y,
        camera_matrix=new_camera_matrix,
    )

#Apply pre runtime computed undistortion map 
def apply_undistortion(
        frame: np.ndarray,
        maps: UndistortionMaps,
) -> np.ndarray:
    """Apply precomputed lens-distortion correction."""

    return cv2.remap(
        frame,
        maps.map_x,
        maps.map_y,
        interpolation=cv2.INTER_LINEAR,
    )