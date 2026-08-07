from __future__ import annotations

import argparse
import sys

import cv2

from tabletop_vision.camera import (
    CameraConfig,
    CameraError,
    Webcam,
)

from tabletop_vision.perception.colour import (
    HSVRange,
    create_hsv_mask,
)

from tabletop_vision.perception.morphology import (
    clean_mask,
)

FRAME_WINDOW = "Original"
MASK_WINDOW = "HSV Mask"
CONTROLS_WINDOW = "HSV Controls"

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactively inspect HSV colour segmentation."
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

    return parser.parse_args()

def do_nothing(_: int) -> None:
    """Trackbar callback required by OpenCV."""
    return None

def create_controls() -> None:
    cv2.namedWindow(
        CONTROLS_WINDOW,
        cv2.WINDOW_NORMAL,
    )

    cv2.createTrackbar(
        "H min",
        CONTROLS_WINDOW,
        0,
        179,
        do_nothing,
    )

    cv2.createTrackbar(
        "H max",
        CONTROLS_WINDOW,
        179,
        179,
        do_nothing,
    )

    cv2.createTrackbar(
        "S min",
        CONTROLS_WINDOW,
        0,
        255,
        do_nothing,
    )

    cv2.createTrackbar(
        "S max",
        CONTROLS_WINDOW,
        255,
        255,
        do_nothing,
    )

    cv2.createTrackbar(
        "V min",
        CONTROLS_WINDOW,
        0,
        255,
        do_nothing,
    )

    cv2.createTrackbar(
        "V max",
        CONTROLS_WINDOW,
        255,
        255,
        do_nothing,
    )

def read_colour_range() -> HSVRange:
    return HSVRange(
        hue_min=cv2.getTrackbarPos(
            "H min",
            CONTROLS_WINDOW,
        ),
        hue_max=cv2.getTrackbarPos(
            "H max",
            CONTROLS_WINDOW,
        ),
        saturation_min=cv2.getTrackbarPos(
            "S min",
            CONTROLS_WINDOW,
        ),
        saturation_max=cv2.getTrackbarPos(
            "S max",
            CONTROLS_WINDOW,
        ),
        value_min=cv2.getTrackbarPos(
            "V min",
            CONTROLS_WINDOW
        ),
        value_max=cv2.getTrackbarPos(
            "V max",
            CONTROLS_WINDOW,
        ),
    )

def run(arguments: argparse.Namespace) -> None:
    config = CameraConfig(
        index=arguments.camera,
        width=arguments.width,
        height=arguments.height,
        fps=arguments.fps,
    )

    create_controls()

    try:
        with Webcam(config) as camera:
            while True:
                frame = camera.read()

                colour_range = read_colour_range()

                mask = create_hsv_mask(
                    frame,
                    colour_range,
                )

                # cleaned_mask = clean_mask(
                #     mask,
                #     kernel_size=3,
                # )

                cv2.imshow(
                    FRAME_WINDOW,
                    frame,
                )

                cv2.imshow(
                    MASK_WINDOW,
                    mask,
                )

                # cv2.imshow(
                #     "Cleaned Mask",
                #     cleaned_mask,
                # )

                key = cv2.waitKey(10) & 0xFF

                if key in (ord("q"),ord("Q"),27,):
                    break

                try:
                    visible = cv2.getWindowProperty(
                        FRAME_WINDOW,
                        cv2.WND_PROP_VISIBLE,
                    )

                    if visible < 1:
                        break

                except cv2.error:
                    break
    finally:
        cv2.destroyAllWindows()
        cv2.waitKey(1)


def main() -> int:
    arguments = parse_arguments()

    try:
        run(arguments)
    except(
        CameraError,
        ValueError,
    ) as error:
        print(
            f"Error: {error}",
            file=sys.stderr,
        )

        return 1
    except KeyboardInterrupt:
        return 130

    return 0

if __name__ == "__main__":
    raise SystemExit(main())