from pathlib import Path

import pytest

from tabletop_vision.learned import (
    SegmentationTrainingConfig,
    validate_training_config,
)

#invalid epochs
def test_training_config_rejects_invalid_epochs(
    tmp_path,
) -> None:
    with pytest.raises(ValueError):
        SegmentationTrainingConfig(
            dataset_yaml=(
                tmp_path / "dataset.yaml"
            ),
            epochs=0,
        )

#invalid image size
def test_training_config_rejects_invalid_image_size(
    tmp_path,
) -> None:
    with pytest.raises(ValueError):
        SegmentationTrainingConfig(
            dataset_yaml=(
                tmp_path / "dataset.yaml"
            ),
            image_size=0,
        )

#invalid batch size
def test_training_config_rejects_invalid_batch_size(
    tmp_path,
) -> None:
    with pytest.raises(ValueError):
        SegmentationTrainingConfig(
            dataset_yaml=(
                tmp_path / "dataset.yaml"
            ),
            batch_size=0,
        )

#missing dataset
def test_training_validation_rejects_missing_dataset(
    tmp_path,
) -> None:
    config = SegmentationTrainingConfig(
        dataset_yaml=(
            tmp_path / "missing.yaml"
        ),
    )

    with pytest.raises(
        FileNotFoundError
    ):
        validate_training_config(
            config
        )

#valid dataset 
def test_training_validation_accepts_dataset_yaml(
    tmp_path,
) -> None:
    dataset_yaml = (
        tmp_path / "dataset.yaml"
    )

    dataset_yaml.write_text(
        "names:\n"
        "  0: target_object\n",
        encoding="utf-8",
    )

    config = SegmentationTrainingConfig(
        dataset_yaml=dataset_yaml,
    )

    validate_training_config(
        config
    )