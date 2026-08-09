from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from tabletop_vision.dataset.models import (
    DatasetImageMetadata,
    DatasetSplit,
)

def create_session_split(
        metadata: Sequence[DatasetImageMetadata],
        train_fraction: float = 0.7,
        validation_fraction: float = 0.15,
        test_fraction: float = 0.15,
        seed: int = 42,
) -> DatasetSplit:
    """Split images while keeping capture sessions together."""

    if not metadata:
        raise ValueError(
            "Cannot split an empty dataset."
        )

    total_fraction = (
        train_fraction
        + validation_fraction
        + test_fraction
    )

    if not math.isclose(
        total_fraction,
        1.0,
        abs_tol=1e-9,
    ): 
        raise ValueError(
            "Split fractions must sum to 1.0."
        )

    if (
        train_fraction <= 0.0
        or validation_fraction <= 0.0
        or test_fraction <= 0.0
    ):
        raise ValueError(
            "All split fractions must be positive."
        )

    sessions: dict[str, list[str]] = (
        defaultdict(list)
    )

    for item in metadata:
        if item.session is None:
            raise ValueError(
                f"{item.filename} has no capture session."
            )
        sessions[item.session].append(
            item.filename
        )

    session_names = list(
        sessions.keys()
    )

    if len(session_names) < 3:
        raise ValueError(
            "At least three capture sessions are requried "
            "for train/validation/test splitting."
        )

    rng = random.Random(seed)

    rng.shuffle(
        session_names
    )

    session_count = len(
        session_names
    )

    train_count = max(
        1,
        int(
            session_count
            * train_fraction
        ),
    )

    validation_count = max(
        1,
        int(
            session_count
            * train_fraction
        ), 
    )

    validation_count = max(
        1,
        int(
            session_count
            * validation_fraction
        ),
    )

    #ALways reserve at least one session for testing

    if (
        train_count
        + validation_count
        >= session_count
    ):
        train_count = (
            session_count
            - validation_count
            - 1
        )

    train_sessions = session_names[
        :train_count
    ]

    validation_sessions = session_names[
        train_count: train_count + validation_count
    ]

    test_sessions = session_names[
        train_count + validation_count:
    ]

    return DatasetSplit(
        train = _filenames_for_sessions(
            sessions,
            train_sessions,
        ),
        validation=_filenames_for_sessions(
            sessions,
            validation_sessions,
        ),
        test = _filenames_for_sessions(
            sessions,
            test_sessions,
        ),
    )

def save_dataset_split(
        split: DatasetSplit,
        output_directory: Path,
) -> None:
    """Write tran, validation and test filename manifests."""

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    _write_manifest(
        output_directory / "train.txt",
        split.train,
    )

    _write_manifest(
        output_directory / "validation.txt",
        split.validation
    )

    _write_manifest(
        output_directory /"test.txt",
        split.test,
    )

def _write_manifest(
        output_path: Path,
        filenames: Sequence[str],
) -> None: 
    output_path.write_text(
        "".join(
            f"{filename}\n"
            for filename in filenames
        ),
        encoding="utf-8",
    )

def _filenames_for_sessions(
    sessions: dict[str, list[str]],
    session_names: Sequence[str],
) -> tuple[str, ...]:
    filenames: list[str] = []

    for session_name in session_names:
        filenames.extend(
            sessions[session_name]
        )
    return tuple(
        sorted(filenames)
    )