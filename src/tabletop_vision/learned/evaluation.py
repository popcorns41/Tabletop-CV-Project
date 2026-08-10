from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ultralytics import YOLO
from ultralytics.utils.metrics import (
    SegmentMetrics,
)

EvaluationSplit = Literal[
    "val",
    "test",
]

@dataclass(frozen=True, slots=True)
class SegmentationEvaluationConfig:
    model_path: Path
    dataset_yaml: Path

    split: EvaluationSplit = "val"

    image_size: int = 640
    batch_size: int = 8

    device: str | int | None = None

    project_directory: Path = Path(
        "runs/evaluation"
    )

    run_name: str = "tabletop_evaluation"

    def __post_init__(self) -> None:
        if self.image_size <= 0:
            raise ValueError(
                "image_size must be positive."
            )

        if self.batch_size <= 0:
            raise ValueError(
                "batch_size must be positive."
            )

        if not self.run_name.strip():
            raise ValueError(
                "run_name must not be empty."
            )

@dataclass(frozen=True, slots=True)
class SegmentationEvaluationReport:
    split: EvaluationSplit

    mask_precision: float
    mask_recall: float

    mask_map_50: float
    mask_map_75: float
    mask_map_50_95: float

    box_map_50: float
    box_map_50_95: float


def validate_evaluation_config(
        config: SegmentationEvaluationConfig,
) -> None:
    if not config.model_path.exists():
        raise FileNotFoundError(
            f"Model does not exist: "
            f"{config.dataset_yaml}"
        )

    
    if not config.dataset_yaml.exists():
        raise FileNotFoundError(
            f"Dataset YAML does not exist: "
            f"{config.dataset_yaml}"
        )

def _create_evaluation_report(
        metrics: SegmentMetrics,
        split: EvaluationSplit,
) -> SegmentationEvaluationReport:
    return SegmentationEvaluationReport(
        split=split,

        mask_precision=float(
            metrics.seg.mp
        ),

        mask_recall=float(
            metrics.seg.map50
        ),

        mask_map_50=float(
            metrics.seg.map50
        ),

        mask_map_75=float(
            metrics.seg.map75
        ),

        mask_map_50_95=float(
            metrics.seg.map
        ),

        box_map_50=float(
            metrics.box.map
        ),

        box_map_50_95=float(
            metrics.box.map
        ),
    )

def evaluate_instance_segmenter(
        config: SegmentationEvaluationConfig,
) -> SegmentationEvaluationReport:
    validate_evaluation_config(
        config
    )

    model = YOLO(
        str(config.model_path)
    )

    if config.device is None:
        metrics = model.val(
            data=str(config.dataset_yaml),
            split=config.split,
            imgsz=config.image_size,
            batch=config.batch_size,
            project=str(
                config.project_directory
            ),
            name=config.run_name,
            plots=True,
        )
    else:
        metrics = model.val(
            data=str(config.dataset_yaml),
            split=config.split,
            imgsz=config.image_size,
            batch=config.batch_size,
            device=config.device,
            project=str(
                config.project_directory
            ),
            name=config.run_name,
            plots=True,
        )

    if not isinstance(
        metrics,
        SegmentMetrics,
    ):
        raise RuntimeError(
            "Expected segmentation metrics "
            "from the model."
        )

    return _create_evaluation_report(
        metrics,
        config.split,
    )