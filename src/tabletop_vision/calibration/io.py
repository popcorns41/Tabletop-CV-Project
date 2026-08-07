from tabletop_vision.calibration.models import (
    CameraCalibrationResult,
    CalibrationObservation,
    CharucoBoardSpec
)

from pathlib import Path
from typing import Sequence
import json

def save_camera_calibration(
        result: CameraCalibrationResult,
        observations: Sequence[CalibrationObservation],
        spec: CharucoBoardSpec,
        output_path: Path,
) -> None:
    """Save calibration parameters and evaluation data as JSON."""

    if len(observations) != len(result.per_view_errors):
        raise ValueError(
            "Observation and per-view error counts do not match."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "model": "opencv_pinhole_radial_tangential",
        "image_width": result.image_size[0],
        "image_height": result.image_size[1],
        "rms_reprojection_error": (
            result.rms_reprojection_error
        ),
        "camera_matrix":(
            result.camera_matrix.tolist()
        ),
        "distortion_coefficients":(
            result.distortion_coefficients
            .reshape(-1)
            .tolist()
        ),
        "board": {
            "squares_x": spec.squares_x,
            "squares_y": spec.squares_y,
            "square_length_mm": spec.square_length_mm,
            "marker_length_mm": spec.marker_length_mm,
            "dictionary": spec.dictionary_name,
        },
        "views": [
            {
                "image": str(observation.image_path),
                "point_count": observation.point_count,
                "reprojection_rmse_pixels": error,
            }
            for observation, error in zip(
                observations,
                result.per_view_errors,
                strict=True,
            )
        ],
    }

    output_path.write_text(
        json.dumps(
            payload,
            indent=2,
        )
        +"\n",
        encoding="utf-8",
    )