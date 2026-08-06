from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from tabletop_vision.calibration import (
    CharucoBoardSpec,
    CharucoDetection,
    create_charuco_detector,
    detect_charuco_board,
)

from tabletop_vision.camera import (
    CameraConfig,
    CameraError,
    Webcam,
)

WINDOW_NAME = "ChArUco Calibration Capture"

DEFAULT_OUTPUT_DIRECTORY = Path("data/calibration/images")

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "detect a ChArUco board and capture camera "
            "calibration images."
        )
    )

    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index. Default: 0",
    )

    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Requested frame width. Default: 1280",
    )

    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Requested frame height. Default: 720"
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Requested frame rate. Default: 30",
    )

    parser.add_argument(
        "--minimum-corners",
        type=int,
        default=8,
        help=(
            "Minimum detected ChArUco corners required "
            "before an image may be saved. Default: 8"
        ),
    )

    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help=(
            "Directory for calibration images. "
            "Default: data/calibration/images"
        ),
    )

    return parser.parse_args()

def create_timestamp() -> str:
    return datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )[:-3]

def draw_text(
        frame: np.ndarray,
        text: str,
        position: tuple[int, int],
        foreground: tuple[int, int, int] = (255, 255, 255),
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.65
    x,y = position

    cv2.putText(
        frame,
        text,
        (x, y),
        font,
        font_scale,
        foreground,
        1,
        cv2.LINE_AA,
    )

def draw_detection(
        frame: np.ndarray,
        detection: CharucoDetection,
) -> None:
    if detection.marker_ids is not None:
        cv2.aruco.drawDetectedMarkers(
            frame,
            detection.marker_corners,
            detection.marker_ids,
        )

    if ( detection.charuco_corners is not None 
        and detection.charuco_ids is not None
        ):
        cv2.aruco.drawDetectedCornersCharuco(
            frame,
            detection.charuco_corners,
            detection.charuco_ids,
            (0, 255, 0),
        )

def calculate_sharpness(frame: np.ndarray) -> float:
    """Return a simple relative sharpness score."""

    grayscale = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY,
    )

    laplacian = cv2.Laplacian(
        grayscale,
        cv2.CV_64F,
    )

    return float(laplacian.var())

def save_frame(
        frame: np.ndarray,
        output_directory: Path,
        image_number: int
) -> Path:

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )
    
    output_path = output_directory / (
        f"calibration_"
        f"{image_number:03d}_"
        f"{create_timestamp()}.png"
    )

    successful = cv2.imwrite(
        str(output_path),
        frame,
    )

    if not successful:
        raise RuntimeError(
            f"Could not save calibration image to {output_path}."
        )

    return output_path

def run(arguments: argparse.Namespace) -> None:
    if arguments.minimum_corners < 4:
        raise ValueError(
            "minimum-corners must be at least 4."
        )

    spec = CharucoBoardSpec()
    detector = create_charuco_detector(spec)

    camera_config = CameraConfig(
        index=arguments.camera,
        width=arguments.width,
        height=arguments.height,
        fps=arguments.fps,
    )

    arguments.output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    saved_image_count = len(
        list(arguments.output_directory.glob("*.png"))
    )

    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL,
    )

    try:
        with Webcam(camera_config) as camera:
            properties = camera.properties()

            print("Calibration capture started")
            print(
                f"  Resolution: "
                f"{properties.width}x{properties.height}"
            )
            print(
                f"  Minimum corners: "
                f"{arguments.minimum_corners}"
            )
            print(
                f"  Existing images: "
                f"{saved_image_count}"
            )
            print()
            print("Controls")
            print("  S: save frame when detection is valid")
            print("  Q or Escape: quit")
            print("  Closing the preview window also quits")

            while True:
                raw_frame = camera.read()

                detection = detect_charuco_board(
                    detector,
                    raw_frame,
                )

                sharpness = calculate_sharpness(
                    raw_frame
                )

                ready_to_save = (
                    detection.corner_count
                    >= arguments.minimum_corners
                )

                display_frame = raw_frame.copy()

                draw_detection(
                    display_frame,
                    detection,
                )

                status_colour = (
                    (0, 255, 0)
                    if ready_to_save
                    else (0, 165, 255)
                )

                status_text = (
                    "READY - press S"
                    if ready_to_save
                    else "Move board into view"
                )

                lines = [
                    f"Markers: {detection.marker_count}",
                    (
                        f"ChArUco corners: "
                        f"{detection.corner_count}/"
                        f"{spec.internal_corner_count}"
                    ),
                    f"Sharpness score: {sharpness:.1f}",
                    f"Saved images: {saved_image_count}",
                ]

                for index, line in enumerate(lines):
                    draw_text(
                        display_frame,
                        line,
                        (12, 30 + index * 30),
                    )

                draw_text(
                    display_frame,
                    status_text,
                    (12, 30 + len(lines) * 30),
                    status_colour,
                )

                cv2.imshow(
                    WINDOW_NAME,
                    display_frame,
                )

                # waitKey also processes the OpenCV window's GUI events.
                raw_key = cv2.waitKey(10)
                key = raw_key & 0xFF

                if key in (
                    ord("q"),
                    ord("Q"),
                    27,  # Escape
                ):
                    print("Capture stopped by user.")
                    break

                # Stop when the macOS red close button is pressed.
                try:
                    window_visible = cv2.getWindowProperty(
                        WINDOW_NAME,
                        cv2.WND_PROP_VISIBLE,
                    )

                    if window_visible < 1:
                        print("Capture window closed.")
                        break

                except cv2.error:
                    # The window may already have been destroyed.
                    break

                if key in (
                    ord("s"),
                    ord("S"),
                ):
                    if not ready_to_save:
                        print(
                            "Frame rejected: only "
                            f"{detection.corner_count} "
                            "corners detected."
                        )
                        continue

                    saved_image_count += 1

                    output_path = save_frame(
                        raw_frame,
                        arguments.output_directory,
                        saved_image_count,
                    )

                    print(
                        f"Saved image {saved_image_count}: "
                        f"{output_path} "
                        f"({detection.corner_count} corners, "
                        f"sharpness {sharpness:.1f})"
                    )

    finally:
        # Runs whether we quit normally, press Ctrl+C,
        # or encounter an exception.
        cv2.destroyAllWindows()

        # Give HighGUI one final opportunity to process closure.
        cv2.waitKey(1)

def main() -> int:
    arguments = parse_arguments()

    try:
        run(arguments)

    except(
        CameraError,
        RuntimeError,
        ValueError,
    ) as error:
        print(f"Error: {error}")
        return 1

    except KeyboardInterrupt:
        print("\nCapture interrupted.")
        return 130

    return 0

if __name__ == "__main__":
    raise SystemExit(main())