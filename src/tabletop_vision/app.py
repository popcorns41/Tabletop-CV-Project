from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from tabletop_vision.camera import CameraConfig, CameraError, CameraProperties, Webcam
from tabletop_vision.metrics import RollingFps
from tabletop_vision.recording import (
    RecordingError,
    VideoRecorder,
    save_snapshot
)

WINDOW_NAME = "Tabletop Vision - Camera Preview"

CAPTURE_DIRECTORY = Path("data/captures")
RECORDING_DIRECTORY = Path("data/recordings")

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Display and record frames from a webcam."
    )

    parser.add_argument(
        "--camera",
        type = int,
        default = 0,
        help = "Camera device index. Default: 0",
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
        help="Requested frame height. Default: 720",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Requested camera FPS. Default: 30",
    )

    return parser.parse_args()

def print_camera_properties(config: CameraConfig, properties: CameraProperties) -> None:
    print("Camera opened successfully")
    print(f"  Index:                {config.index}")
    print(f"  Backend:              {properties.backend}")
    print(f"  Requested resolution: {config.width}x{config.height}")
    print(f"  Actual resolution:    {properties.width}x{properties.height}")
    print(f"  Requested FPS:        {config.fps}")
    print(f"  Reported FPS:         {properties.fps:.2f}")
    print()
    print("Controls")
    print("  Q or Escape: quit")
    print("  S: save raw snapshot")
    print("  R: start or stop raw video recording")

def draw_text(
        frame: np.ndarray,
        text: str,
        position: tuple[int, int],
) -> None:
    font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX

    #Dark outline keeps the text readable against bright camera frames.
    cv2.putText(
        frame,
        text,
        position,
        font,
        0.65,
        (0,0,0),
        4,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        text,
        position,
        font,
        0.65,
        (255,255,255),
        1,
        cv2.LINE_AA,
    )

def draw_diagnostics(
        frame: np.ndarray,
        measured_fps: float,
        is_recording: bool,
) -> None:
    height, width = frame.shape[:2]

    lines = [
        f"Resolution: {width}x{height}",
        f"Measured FPS: {measured_fps:.1f}",
        f"Recording: {'ON' if is_recording else 'Off'}",
        "Q: quit | S: snapshot | R: record",
    ]

    for index, line in enumerate(lines):
        y_position = 30 + index * 28
        draw_text(frame, line, (10, y_position))

def run(config: CameraConfig) -> None:
    recorder = VideoRecorder(RECORDING_DIRECTORY)
    fps_counter = RollingFps(window_size=30)

    with Webcam(config) as camera:
        properties = camera.properties()
        print_camera_properties(config, properties)

        # writer_fps is the FPS used for the video writer. 
        # If the camera reports a valid FPS, we use that. 
        # Otherwise, we fall back to the requested FPS.
        writer_fps = properties.fps if properties.fps > 1.0 else float(config.fps)

        try:
            while True:
                raw_frame = camera.read()
                measured_fps = fps_counter.update()

                if recorder.is_recording:
                    recorder.write(raw_frame)

                # Overlays are drawn only on a copy used for display.
                display_frame = raw_frame.copy()

                draw_diagnostics(
                    display_frame,
                    measured_fps,
                    recorder.is_recording
                )

                cv2.imshow(WINDOW_NAME, display_frame)

                key = cv2.waitKey(1) & 0xFF

                if key in (ord("q"),27):
                    break

                if key == ord("s"):
                    path = save_snapshot(raw_frame, CAPTURE_DIRECTORY)
                    print(f"Snapshot saved: {path}")

                if key == ord("r"):
                    if recorder.is_recording:
                        path = recorder.stop()
                        print(f"Recording saved: {path}")
                    else:
                        path = recorder.start(raw_frame,writer_fps)
                        print(f"Recording started: {path}")
        finally:
            completed_path = recorder.stop()

            if completed_path is not None:
                print(f"Recording saved: {completed_path}")

            cv2.destroyAllWindows()

def main() -> int:
    arguments = parse_arguments()

    config = CameraConfig(
        index=arguments.camera,
        width=arguments.width,
        height=arguments.height,
        fps=arguments.fps,
    )

    try:
        run(config)
    except (CameraError,RecordingError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nCapture interrupted.")
        return 130
    
    return 0