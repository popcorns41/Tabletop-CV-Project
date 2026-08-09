from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from enum import Enum, auto

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

class UserAction(Enum):
    NONE = auto()
    SAVE_POSITIVE = auto()
    SAVE_NEGATIVE = auto()
    QUIT = auto()

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

    parser.add_argument(
        "--session",
        type=str,
        default=None,
        help="Optional capture-session identifier.",
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
        "P: save positive frame",
        "N: save negative frame",
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

# ==== User Action ====


def read_user_action() -> UserAction:
    """Return quit and save requests from the OpenCV UI."""

    key = cv2.waitKey(10) & 0xFF

    if key in (
        ord("q"),
        ord("Q"),
        27,
    ):
        return UserAction.QUIT

    if key in (
        ord("p"),
        ord("P"),
    ):
        return UserAction.SAVE_POSITIVE

    if key in (
        ord("n"),
        ord("N"),
    ):
        return UserAction.SAVE_NEGATIVE

    try:
        visible = cv2.getWindowProperty(
            WINDOW_NAME,
            cv2.WND_PROP_VISIBLE,
        )

        if visible < 1:
            return UserAction.QUIT

    except cv2.error:
        return UserAction.QUIT

    return UserAction.NONE


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
        session=arguments.session,
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

                action = read_user_action()

                if action is UserAction.QUIT:
                    break

                if action is UserAction.SAVE_POSITIVE:
                    metadata = writer.save_frame(
                        frame,
                        target_present=True,
                    )

                    print(
                        f"Saved positive: {metadata.filename}"
                    )

                elif action is UserAction.SAVE_NEGATIVE:
                    metadata = writer.save_frame(
                        frame,
                        target_present=False,
                    )

                    print(
                        f"Saved positive: {metadata.filename}"
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