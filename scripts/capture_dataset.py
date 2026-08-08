from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from tabletop_vision.camera import(
    CameraConfig,
    CameraError,
    Webcam,
)

from tabletop_vision.dataset import(
    DatasetWriter,
)

WINDOW_NAME = "Dataset Capture"

DEFAULT_OUTPUT_DIRECTORY = Path(
    "data/dataset"
)

# ===== Argument Parser =====

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Capture raw webcam frames for "
            "the tabletop vision dataset."
        )
    )

    parser.add_argument(
        "--camera",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--width",
        type=int,
        default=1280,
    )

    parser.add_argument(
        "--height",
        type=int,
        default=720,
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
    )

    parser.add_argument(
        "--environment",
        type=str,
        default=None,
        help=(
            "Optional capture environment label, "
            "for example 'desk_daylight'."
        ),
    )

    return parser.parse_args()


# ==== Draw Status ====

def draw_status(
        frame: np.ndarray,
        image_count: int,
        environment: str | None,
) -> None:
    lines = [
        f"Saved images: {image_count}",
        (
            f"Environment: "
            f"{environment or 'unspecified'}"
        ),
        "S: save frame",
        "Q / Esc: quit"
    ]

    for index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (
                12,
                30 + index * 28,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

def read_user_action() -> tuple[bool, bool]:
    """Return quit and save requests from the OpenCV UI."""

    key = cv2.waitKey(10) & 0xFF

    if key in (
        ord("q"),
        ord("Q"),
        27,
    ):
        return True, False

    try:
        visible = cv2.getWindowProperty(
            WINDOW_NAME,
            cv2.WND_PROP_VISIBLE,
        )

        if visible < 1:
            return True, False

    except cv2.error:
        return True, False

    save_requested = key in (
        ord("s"),
        ord("S"),
    )

    return False, save_requested


def run(
    arguments: argparse.Namespace,
) -> None:
    config = CameraConfig(
        index=arguments.camera,
        width=arguments.width,
        height=arguments.height,
        fps=arguments.fps,
    )

    writer = DatasetWriter(
        root_directory=arguments.output,
        environment=arguments.environment,
    )

    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL,
    )

    try:
        with Webcam(config) as camera:
            while True:
                frame = camera.read()

                display_frame = frame.copy()

                draw_status(
                    display_frame,
                    writer.image_count,
                    arguments.environment,
                )

                cv2.imshow(
                    WINDOW_NAME,
                    display_frame,
                )

                (
                    quit_requested,
                    save_requested,
                ) = read_user_action()

                if quit_requested:
                    break

                if save_requested:
                    metadata = writer.save_frame(
                        frame
                    )

                    print(
                        f"Saved {metadata.filename}"
                    )

    finally:
        cv2.destroyAllWindows()
        cv2.waitKey(1)

def main() -> int:
    arguments = parse_arguments()

    try:
        run(arguments)

    except (
        CameraError,
        RuntimeError,
        ValueError,
    ) as error:
        print(
            f"Dataset capture failed: {error}",
            file=sys.stderr,
        )

        return 1

    except KeyboardInterrupt:
        return 130

    return 0


if __name__ == "__main__":
    raise SystemExit(main())