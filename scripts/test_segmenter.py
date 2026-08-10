from __future__ import annotations

import argparse
import sys

import cv2

from tabletop_vision.camera import (
    CameraConfig,
    CameraError,
    Webcam,
)
from tabletop_vision.learned import(
    InstanceSegmenter,
)

WINDOW_NAME = "Pretrained Segmentation"

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--camera",
        type=int,
        default=0
    )

    return parser.parse_args()

def run(
        arguments: argparse.Namespace,
) -> None:
    segmenter = InstanceSegmenter(
        "yolo26n-seg.pt",
        confidence_threshold=0.5,
    )

    config = CameraConfig(
        index=arguments.camera,
        width=1280,
        height=720,
        fps=30,
    )

    try:
        with Webcam(config) as camera:
            while True:
                frame = camera.read()

                predictions = segmenter.predict(
                    frame
                )

                display_frame = frame.copy()

                for prediction in predictions:
                    polygon = prediction.polygon.astype(
                        "int32"
                    )

                    cv2.polylines(
                        display_frame,
                        [polygon],
                        isClosed=True,
                        color=(0, 255, 0),
                        thickness=2,
                    )

                    if len(polygon) > 0:
                        x, y = polygon[0]

                        cv2.putText(
                            display_frame,
                            (
                                f"{prediction.class_name} "
                                f"{prediction.confidence:.2f}"
                            ),
                            (int(x), int(y)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0),
                            2,
                            cv2.LINE_AA,
                        )

                cv2.imshow(
                    WINDOW_NAME,
                    display_frame,
                )

                key = cv2.waitKey(10) & 0xFF

                if key in (
                    ord("q"),
                    ord("Q"),
                    27,
                ):
                    break

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
            f"Segmentation failed: {error}",
            file=sys.stderr,
        )
        return 1

    except KeyboardInterrupt:
        return 130

    return 0


if __name__ == "__main__":
    raise SystemExit(main())