from __future__ import annotations

class ExponentialSmoother:
    """Exponentially smooth a sequence of scalar measurements."""

    def __init__(
            self,
            alpha: float,
    ) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError(
                "alpha must lie in the interval (0, 1]."
            )

        self._alpha = alpha
        self._value: float | None = None

    @property
    def value(self) -> float | None:
        return self._value

    def update(
            self,
            measurement:float,
    ) -> float:
        if self._value is None:
            self._value = measurement
            return self._value

        self._value = (
            self._alpha * measurement
            + (1.0 - self._alpha) * self._value
        )

        return self._value

    def reset(self) -> None:
        self._value = None


class PositionSmoother:
    """Exponentially smooth 2D image-space positions."""

    def __init__(
            self,
            alpha: float,
    ) -> None:
        self._x = ExponentialSmoother(alpha)
        self._y = ExponentialSmoother(alpha)

    def update(
            self,
            measurement: tuple[int, int],
    ) -> tuple[float, float]:
        measure_x, measure_y = measurement

        filtered_x = self._x.update(
            float(measure_x)
        )

        filtered_y = self._y.update(
            float(measure_y)
        )

        return filtered_x, filtered_y

    def reset(self) -> None:
        self._x.reset()
        self._y.reset()