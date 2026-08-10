from tabletop_vision.learned.evaluation import(
    SegmentationEvaluationConfig,
    validate_evaluation_config
)

import pytest

def test_evaluation_config_rejects_invalid_image_size(
        tmp_path,
) -> None:
    with pytest.raises(ValueError):
        SegmentationEvaluationConfig(
            model_path=tmp_path / "model.pt",
            dataset_yaml=tmp_path / "dataset.yaml",
            image_size=0,
        )

def test_evaluation_rejects_missing_model(
        tmp_path,
) -> None:
    dataset_yaml = (
        tmp_path / "dataset.yaml"
    )

    dataset_yaml.touch()

    config = SegmentationEvaluationConfig(
        model_path=(
            tmp_path / "missing.pt"
        ),
        dataset_yaml=dataset_yaml,
    )

    with pytest.raises(
        FileNotFoundError
    ):
        validate_evaluation_config(
            config
        )

def test_evaluation_accepts_existing_resources(
    tmp_path,
) -> None:
    model = tmp_path / "model.pt"
    dataset_yaml = (
        tmp_path / "dataset.yaml"
    )

    model.touch()
    dataset_yaml.touch()

    config = SegmentationEvaluationConfig(
        model_path=model,
        dataset_yaml=dataset_yaml,
    )

    validate_evaluation_config(
        config
    )