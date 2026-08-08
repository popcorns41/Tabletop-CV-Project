import json

import numpy as np

from tabletop_vision.dataset import (
    DatasetWriter,
)

def test_dataset_writer_saves_image(
        tmp_path,
) -> None:
    writer = DatasetWriter(
        root_directory=tmp_path,
        environment="test_room",
    )

    frame = np.zeros(
        (100, 200, 3),
        dtype=np.uint8,
    )

    metadata = writer.save_frame(
        frame
    )

    image_path = (
        tmp_path
        / "images"
        / metadata.filename
    )

    assert image_path.exists()


    assert metadata.width == 200
    assert metadata.height == 100

    assert metadata.environments == (
        "test_room"
    )

def test_dataset_writer_appends_metadata(
        tmp_path,
) -> None:
    writer = DatasetWriter(
        root_directory=tmp_path,
    )

    frame= np.zeros(
        (50, 80, 3),
        dtype=np.uint8,
    )

    writer.save_frame(frame)
    writer.save_frame(frame)

    metadata_path = (
        tmp_path / "metadata.jsonl"
    )

    lines = metadata_path.read_text(
        encoding="utf-8"
    ).splitlines()

    assert len(lines) == 2

    first = json.loads(
        lines[0]
    )

    second = json.loads(
        lines[1]
    )

    assert first["filename"] == (
        "frame_000001.jpg"
    )

    assert second["filename"] == (
        "frame_000002.jpg"
    )

def test_dataset_writer_continues_existing_indices(
    tmp_path,
) -> None:
    frame = np.zeros(
        (50, 80, 3),
        dtype=np.uint8,
    )

    first_writer = DatasetWriter(
        tmp_path
    )

    first_writer.save_frame(frame)
    first_writer.save_frame(frame)

    # Simulate restarting the application.
    second_writer = DatasetWriter(
        tmp_path
    )

    metadata = second_writer.save_frame(
        frame
    )

    assert metadata.filename == (
        "frame_000003.jpg"
    )