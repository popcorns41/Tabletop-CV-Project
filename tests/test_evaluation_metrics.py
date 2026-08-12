import pytest

from tabletop_vision.evaluation.metrics import (
    orientation_error_degrees,
    positional_error_pixels,
)

def test_positional_error_pixels() -> None:
    error = positional_error_pixels(
        predicted=(503.0, 304.0),
        ground_truth=(500.0, 300.0),
    )

    assert error == pytest.approx(5.0)

def test_positional_error_is_zero_for_exact_match() -> None:
    error = positional_error_pixels(
        predicted=(100.0, 200.0),
        ground_truth=(100.0, 200.0),
    )

    assert error == pytest.approx(
        0.0
    )

def test_orientation_error_degrees() -> None:
    error = orientation_error_degrees(
        predicted=35.0,
        ground_truth=30.0,
    )

    assert error == pytest.approx(
        5.0
    )

def test_orientation_error_handles_axis_symmetry(
) -> None:
    error = orientation_error_degrees(
        predicted=-89.0,
        ground_truth=89.0,
    )

    assert error == pytest.approx(
        2.0
    )

def test_orientation_error_treats_180_as_same_axis(
) -> None:
    error = orientation_error_degrees(
        predicted=10.0,
        ground_truth=190.0,
    )

    assert error == pytest.approx(
        0.0
    )