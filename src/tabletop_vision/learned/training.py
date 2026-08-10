from __future__ import annotations

from dataclasses import dataclass 
from pathlib import Path

from ultralytics import YOLO

@dataclass(frozen=True, slots=True)
class SegmentationTrainingConfig:
    """Configuration for fine-tuning an instance-segmentation model."""

    dataset_yaml: Path

    model: str = "yolo26n-seg.pt"

    epochs: int = 100
    image_size: int = 640
    batch_size: int = 8

    device: str | int | None = None

    project_directory: Path = Path(
        "runs/segmentation"
    )

    run_name: str = "tabletop-segmentation"

    def __post_init__(self) -> None: 
        if self.epochs <= 0:
            raise ValueError(
                "epochs must be postive."
            )

        if self.image_size <= 0:
            raise ValueError(
                "image_size must be postive."
            )

        if self.batch_size <= 0:
            raise ValueError(
                "batch_size must be positive."
            )

        if not self.run_name.strip():
            raise ValueError(
                "run_name must not be empty."
            )

def validate_training_config(
        config: SegmentationTrainingConfig,
) -> None:
    """Validate resource required for model training."""

    if not config.dataset_yaml.exists():
        raise FileNotFoundError(
            f"Dataset YAML does not exist: "
            f"{config.dataset_yaml}"
        )

    if not config.dataset_yaml.is_file():
        raise ValueError(
            f"Dataset YAML is not a file: "
            f"{config.dataset_yaml}"
        )

def train_instance_segmenter(
        config: SegmentationTrainingConfig
) -> None: 
    """Fine-tune a pretrained instance-segmentation model."""

    validate_training_config(
        config
    )

    model = YOLO(
        config.model
    )

    training_arguments = {
        "data": str(config.dataset_yaml),
        "epochs": config.epochs,
        "imgsz": config.image_size,
        "batch": config.batch_size,
        "project": str(
            config.project_directory
        ),
        "name": config.run_name,
    }

    if config.device is not None:
        training_arguments["device"] = (
            config.device
        )

    model.train(
        **training_arguments
    )