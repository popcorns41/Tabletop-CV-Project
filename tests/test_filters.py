import pytest

from tabletop_vision.tracking import (
    ExponentialSmoother,
    PositionSmoother,
)


def test_exponential_smoother_uses_first_measurement_directly() -> None:
    smoother = ExponentialSmoother(
        alpha=0.5
    )

    result = smoother.update(10.0)

    assert result == pytest.approx(10.0)

def test_exponential_smoother_blends_measurements() -> None:
    smoother = ExponentialSmoother(
        alpha=0.5
    )

    result = smoother.update(10.0)

    assert result == pytest.approx(10.0)


def test_exponential_smoother_reset_clears_history() -> None:
    smoother = ExponentialSmoother(
        alpha=0.5
    )

    smoother.update(10.0)
    smoother.update(20.0)

    smoother.reset()

    result = smoother.update(100.0)

    assert result == pytest.approx(100.0)

def test_position_smoother_filters_axes_independently() -> None:
    smoother = PositionSmoother(
        alpha=0.5
    )

    first = smoother.update(
        (100, 200)
    )

    second = smoother.update(
        (120, 180)
    )

    assert first == pytest.approx(
        (100.0, 200.0)
    )

    assert second == pytest.approx(
        (110.0, 190.0)
    )