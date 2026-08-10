from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tabletop_vision.learned import (
    SegmentationTrainingConfig,
    train_instance_segmenter,
)

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune the tabletop instance "
            "segmentation model."
        )
    )

    parser.add_argument(
        "-data",
        type=Path,
        required=True,
        help="Path to the exported dataset.yaml.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="yolo26n-seg.pt",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=640,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--project",
        type=Path,
        default=Path(
            "runs/segmentation"
        ),
    )

    parser.add_argument(
        "--name",
        type=str,
        default="tabletop-segmentation",
    )

    return parser.parse_args()

def run(
        arguments: argparse.Namespace,
) -> None:
    config = SegmentationTrainingConfig(
        dataset_yaml=arguments.data,
        model=arguments.model,
        epochs=arguments.epochs,
        image_size=arguments.image_size,
        batch_size=arguments.batch_size,
        device=arguments.device,
        project_directory=arguments.project,
        run_name=arguments.name,
    )

    print ("starting segmentation training")
    print(
        f"  Model:      {config.model}"
    )
    print(
        f"  Dataset:    {config.dataset_yaml}"
    )
    print(
        f"  Epochs:     {config.epochs}"
    )
    print(
        f"  Image size: {config.image_size}"
    )
    print(
        f"  Batch size: {config.batch_size}"
    )
    print(
        f"  Device:     "
        f"{config.device or 'automatic'}"
    )
    print()

    train_instance_segmenter(
        config
    )

def main() -> int:
    arguments = parse_arguments()

    try:
        run(arguments)

    except (
        FileNotFoundError,
        RuntimeError,
        ValueError,
    ) as error:
        print(
            f"Training failed: {error}",
            file=sys.stderr,
        )
        return 1
    except KeyboardInterrupt:
        return 130

    return 0

if __name__ == "__main__":
    raise SystemExit(main())