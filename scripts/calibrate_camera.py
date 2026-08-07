from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

from tabletop_vision.calibration import(
    CalibrationObservation,
    CharucoBoardSpec,
    calibrate_camera,
    create_calibration_observation,
    create_charuco_board,
    create_charuco_detector,
    detect_charuco_board,
    save_camera_calibration,
)

DEFAULT_IMAGES_DIRECTORY = Path(
    "data/calibration/images"
)

DEFAULT_OUTPUT_PATH = Path(
    "data/calibration/results/camera_calibration.json"
)

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
        "Estimate webcam intrinsics and distortion "
        "from saved ChArUco images."
        )
    )

    parser.add_argument(
        "--images",
        type=Path,
        default=DEFAULT_IMAGES_DIRECTORY,
        help=(
            "Directory containing calibration images. "
            "Default: data/calibration/images"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=(
            "Output calibration JSON path. "
            "Default: data/calibration/result/"
            "camera_calibration.json"
        ),
    )

    parser.add_argument(
        "--minimum-corners",
        type=int,
        default=8,
        help=(
            "Minimum detected corners required per image. "
            "Default: 8"
        ),
    )

    parser.add_argument(
        "--minimum-images",
        type=int,
        default=8,
        help=(
            "Minimum accepted images required. "
            "Default: 10"
        )
    )

    return parser.parse_args()

def find_image_paths(
        directory: Path,
) -> list[Path]:
    supported_suffixes = {
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".tiff",
    }

    return sorted(
        path
        for path in directory.iterdir()
        if (
            path.is_file()
            and path.suffix.lower()
            in supported_suffixes
        )
    )

def collect_observations(
        image_paths: list[Path],
        spec: CharucoBoardSpec,
        minimum_corners:int,
) -> tuple[
    list[CalibrationObservation],
    tuple[int, int],
]:
    board = create_charuco_board(spec)
    detector = create_charuco_detector(spec)

    observations: list[CalibrationObservation] = []
    expected_image_size: tuple[int, int] | None = None

    for image_path in image_paths:
        image = cv2.imread(str(image_path))

        if image is None:
            print(
                f"Skipped unreadable image: {image_path}"
            )
            continue

        height, width = image.shape[:2]
        image_size = (width, height)

        if expected_image_size is None:
            expected_image_size = image_size

        elif image_size != expected_image_size:
            print(
                f"Skipped {image_path.name}: "
                f"resolution {width}x{height} does not match "
                f"{expected_image_size[0]}x"
                f"{expected_image_size[1]}."
            )
            continue

        detection = detect_charuco_board(
            detector,
            image,
        )

        if detection.corner_count < minimum_corners:
            print(
                f"Rejected {image_path.name}: "
                f"{detection.corner_count} corners."
            )
            continue

        observation = create_calibration_observation(
            board=board,
            detection=detection,
            image_path=image_path,
        )

        observations.append(observation)

        print(
            f"Accepted {image_path.name}: "
            f"{observation.point_count} corners."
        )

    if expected_image_size is None:
        raise ValueError(
            "No readable calibration images were found."
        )

    return observations, expected_image_size


def run(arguments: argparse.Namespace) -> None:
    if arguments.minimum_corners < 4:
        raise ValueError(
            "minimum-corners must be at least 4."
        )

    if arguments.minimum_images < 3:
        raise ValueError(
            "minimum-images must be at least 3."
        )

    if not arguments.images.exist():
        raise ValueError(
            f"Image directory does not exist:"
            f"{arguments.images}"
        )

    image_paths = find_image_paths(
        arguments.images
    )

    if not image_paths:
        raise ValueError(
            f"No calibration images found in "
            f"{arguments.images}."
        )

    spec = CharucoBoardSpec()

    observations, image_size = collect_observations(
        image_paths=image_paths,
        spec = spec,
        minimum_corners= arguments.minimum_corners,
    )

    if len(observations) < arguments.minimum_images:
        raise ValueError(
            f"Only {len(observations)} valid images were found; "
            f"{arguments.minimum_images} are required."
        )

    result = calibrate_camera(
        observations=observations,
        image_size=image_size,
    )

    save_camera_calibration(
        result=result,
        observations=observations,
        spec=spec,
        output_path=arguments.output,
    )

    print()
    print("Camera calibration completed")
    print(
        f"  Accepted images:       "
        f"{len(observations)}"
    )
    print(
        f"  Image resolution:      "
        f"{image_size[0]}x{image_size[1]}"
    )
    print(
        f"  RMS reprojection error: "
        f"{result.rms_reprojection_error:.4f} px"
    )

    print()
    print("Camera matrix:")
    print(result.camera_matrix)

    print()
    print("Distortion coefficients:")
    print(
        result.distortion_coefficients.reshape(-1)
    )

    print()
    print("Per-view reprojection RMSE:")

    #ranked views is a list of tuples, 
    # each containing a CalibrationObservation 
    # and its corresponding reprojection error, 
    # sorted in descending order of error

    ranked_views = sorted(
        zip(
            observations,
            result.per_view_errors,
            strict=True
        ),
        key=lambda pair: pair[1],
        reverse=True,
    )

    
    for observation, error in ranked_views:
        print(
            f"  {error:8.4f} px  "
            f"{observation.image_path.name}"
        )

    print()
    print(f"Saved result: {arguments.output}")

def main() -> int:
    arguments = parse_arguments()

    try:
        run(arguments)

    except (
        RuntimeError,
        ValueError,
        cv2.error,
    ) as error:
        print(
            f"Calibration failed: {error}",
            file=sys.stderr,
        )
        return 1
    return 0

if __name__ == "__main__":
    raise SystemExit(main())