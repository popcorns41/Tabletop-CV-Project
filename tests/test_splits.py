from tabletop_vision.dataset import (
    DatasetImageMetadata,
    create_session_split,
)

import pytest

def create_metadata(
        filename: str,
        session: str,
) -> DatasetImageMetadata:
    return DatasetImageMetadata(
        filename=filename,
        width=1280,
        height=720,
        timestamp="2026-01-01T00:00:00+00:00",
        environment="test",
        session=session,
        target_present=True,
    )

def test_session_split_keeps_sessions_together() -> None:
    metadata = [
        create_metadata("a1.jpg", "session_a"),
        create_metadata("a2.jpg", "session_a"),

        create_metadata("b1.jpg", "session_b"),
        create_metadata("b2.jpg", "session_b"),

        create_metadata("c1.jpg", "session_c"),
        create_metadata("c2.jpg", "session_c"),

        create_metadata("d1.jpg", "session_d"),
        create_metadata("d2.jpg", "session_d"),

        create_metadata("e1.jpg", "session_e"),
        create_metadata("e2.jpg", "session_e"),
    ]

    split = create_session_split(
        metadata=metadata,
        seed=42,
    )

    split_sets = [
        set(split.train),
        set(split.validation),
        set(split.test),
    ]

    # Check that each session's images are in the same split

    for session_prefix in (
        "a",
        "b",
        "c",
        "d",
        "e",
    ):
        containing_splits = [
            split_set
            for split_set in split_sets
            if any(
                filename.startswith(
                    session_prefix
                )
                for filename in split_set
            )
        ]

        assert len(containing_splits) == 1

def test_session_split_assigns_every_image_once() -> None:
    metadata = [
        create_metadata(
            f"frame_{index}.jpg",
            f"session_{index // 2}",
        )
        for index in range(10)
    ]

    split = create_session_split(
        metadata=metadata,
        seed=42,
    )

    assigned = (
        list(split.train)
        +list(split.validation)
        +list(split.test)
    )

    expected = [
        item.filename
        for item in metadata
    ]

    assert sorted(assigned) == sorted(
        expected
    )

    assert len(assigned) == len(
        set(assigned)
    )

def test_session_split_is_reproducible() -> None:
    metadata = [
        create_metadata(
            f"frame_{index}.jpg",
            f"session_{index}",
        )
        for index in range(10)
    ]

    first = create_session_split(
        metadata=metadata,
        seed=123,
    )

    second = create_session_split(
        metadata=metadata,
        seed=123,
    )

    assert first == second



def test_session_split_rejects_missing_session() -> None:
    metadata = [
        DatasetImageMetadata(
            filename="frame.jpg",
            width=1280,
            height=720,
            timestamp="2026-01-01T00:00:00+00:00",
            environment="test",
            session=None,
            target_present=True,
        )
    ]

    with pytest.raises(ValueError):
        create_session_split(
            metadata
        )