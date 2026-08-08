from __future__ import annotations

import argparse
import sys

import numpy as np

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

from tabletop_vision.perception.contours import (
    RotatedRectangle,
    long_axis_endpoints,
)

from tabletop_vision.perception.detection import (
    detect_largest_object
)

from tabletop_vision.perception.models import (
    ObjectDetection
)

from tabletop_vision.tracking.filters import (
    PositionSmoother
)

FRAME_WINDOW = "Original"
MASK_WINDOW = "HSV Mask"
CONTROLS_WINDOW = "HSV Controls"

#DEFAULT_HSV_RANGE for Pepsi Deep Blue Can
DEFAULT_HSV_RANGE = HSVRange(
    hue_min=63,          
    hue_max=179,
    saturation_min=59,
    saturation_max=255,
    value_min=15,
    value_max=91,
)

# ========================== SYS ARGS PARSING ==========================

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

# ======================== CAMERA CONFIGURATION =========================

def create_camera_config(
        arguments: argparse.Namespace,
) -> CameraConfig:
    return CameraConfig(
        index=arguments.camera,
        width=arguments.width,
        height=arguments.height,
        fps=arguments.fps,
    )

# ======================== INTERACTIVE HSV MASK INSPECTION =========================

def do_nothing(_: int) -> None:
    """Trackbar callback required by OpenCV."""
    return None

def create_controls(initial_range: HSVRange) -> None:
    cv2.namedWindow(
        CONTROLS_WINDOW,
        cv2.WINDOW_NORMAL,
    )

    cv2.createTrackbar(
        "H min",
        CONTROLS_WINDOW,
        initial_range.hue_min,
        179,
        do_nothing,
    )

    cv2.createTrackbar(
        "H max",
        CONTROLS_WINDOW,
        initial_range.hue_max,
        179,
        do_nothing,
    )

    cv2.createTrackbar(
        "S min",
        CONTROLS_WINDOW,
        initial_range.saturation_min,
        255,
        do_nothing,
    )

    cv2.createTrackbar(
        "S max",
        CONTROLS_WINDOW,
        initial_range.saturation_max,
        255,
        do_nothing,
    )

    cv2.createTrackbar(
        "V min",
        CONTROLS_WINDOW,
        initial_range.value_min,
        255,
        do_nothing,
    )

    cv2.createTrackbar(
        "V max",
        CONTROLS_WINDOW,
        initial_range.value_max,
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

# ======================= DRAW FUNCTIONS =========================

def draw_diagnostic_text(
    frame: np.ndarray,
    detection: ObjectDetection | None,
) -> None:
    if detection is None:
        return

    centre_x, centre_y = detection.centroid

    rectangle = detection.rotated_rectangle

    lines = [
        f"({centre_x}, {centre_y})",
        f"angle = {rectangle.angle_degrees:.1f} deg",
        f"area = {detection.area_pixels_squared:.0f} px^2",
        (
            f"size = "
            f"{rectangle.width:.0f} x "
            f"{rectangle.height:.0f} px"
        ),
    ]

    text_x = centre_x + 12
    text_y = centre_y + 24
    line_spacing = 24

    for index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (
                text_x,
                text_y + index * line_spacing,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
    

def draw_detection(
    frame: np.ndarray,
    detection: ObjectDetection | None,
) -> None:
    if detection is None:
        return

    cv2.drawContours(
        frame,
        [detection.contour],
        -1,
        (0, 255, 0),
        2,
    )

    draw_centroid(
        frame,
        detection.centroid,
    )

    draw_rotated_rectangle(
        frame,
        detection.rotated_rectangle,
    )

    start_point, end_point = long_axis_endpoints(detection.rotated_rectangle)

    draw_long_axis_direction(
        frame,
        start_point=start_point,
        end_point=end_point,
        rotated_rectangle=detection.rotated_rectangle,
    )

    draw_diagnostic_text(
        frame=frame,
        detection=detection,
    )

def draw_long_axis_direction(
        frame: np.ndarray,
        start_point: tuple[int,int],
        end_point: tuple[int,int],
        rotated_rectangle: RotatedRectangle
) -> None:
    cv2.line(
        frame,
        start_point,
        end_point,
        (0,255,255),
        3
    )

def draw_centroid(
    frame: np.ndarray,
    centroid: tuple[int, int],
) -> None:

    cv2.drawMarker(
        frame,
        centroid,
        (0, 0, 255),
        markerType=cv2.MARKER_CROSS,
        markerSize=24,
        thickness=2,
    )

def draw_rotated_rectangle(
    frame: np.ndarray,
    rectangle: RotatedRectangle,
) -> None:
    box_points = rectangle.corners.astype(
        np.int32
    )

    cv2.polylines(
        frame,
        [box_points],
        isClosed=True,
        color=(255, 0, 255),
        thickness=2,
    )

def draw_filtered_position(
    frame: np.ndarray,
    position: tuple[float, float],
) -> None:
    x = int(
        round(position[0])
    )

    y = int(
        round(position[1])
    )

    cv2.drawMarker(
        frame,
        (x, y),
        (255, 255, 0),
        markerType=cv2.MARKER_DIAMOND,
        markerSize=18,
        thickness=2,
    )

# ====================== FRAME PROCESSING =========================

def process_frame(
    frame: np.ndarray,
    position_smoother: PositionSmoother,
) -> tuple[np.ndarray, np.ndarray]:
    colour_range = read_colour_range()

    mask = create_hsv_mask(
        frame,
        colour_range,
    )

    detection = detect_largest_object(
        mask,
        minimum_area=1000.0,
    )

    display_frame = frame.copy()


    draw_detection(
        display_frame,
        detection,
    )

    if detection is not None:
        filtered_position = position_smoother.update(
            detection.centroid
        )

        draw_filtered_position(
            display_frame,
            filtered_position,
        )

    return display_frame, mask


def show_windows(
    display_frame: np.ndarray,
    mask: np.ndarray,
) -> None:
    cv2.imshow(
        FRAME_WINDOW,
        display_frame,
    )

    cv2.imshow(
        MASK_WINDOW,
        mask,
    )

def is_window_closed() -> bool:
    try:
        visible = cv2.getWindowProperty(
            FRAME_WINDOW,
            cv2.WND_PROP_VISIBLE,
        )

        return visible < 1

    except cv2.error:
        return True

def quit_key_pressed() -> bool:
    key = cv2.waitKey(10) & 0xFF

    return key in (
        ord("q"),
        ord("Q"),
        27,
    )

def should_quit() -> bool:
    return (
        quit_key_pressed()
        or is_window_closed()
    )

def run(
    arguments: argparse.Namespace,
) -> None:
    config = create_camera_config(
        arguments
    )

    create_controls(
        DEFAULT_HSV_RANGE
    )

    try:
        with Webcam(config) as camera:
            while True:
                frame = camera.read()

                position_smoother = PositionSmoother(
                    alpha=0.25
                )

                display_frame, mask = process_frame(
                    frame,
                    position_smoother,
                )

                show_windows(
                    display_frame,
                    mask,
                )

                if should_quit():
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