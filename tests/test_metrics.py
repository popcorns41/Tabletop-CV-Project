import pytest

from tabletop_vision.metrics import RollingFps

def test_rolling_fps_calculates_frame_rate() -> None:
    counter = RollingFps(window_size=4)

    assert counter.update(0.0) == 0.0
    assert counter.update(0.1) == pytest.approx(10.0)
    assert counter.update(0.2) == pytest.approx(10.0)
    assert counter.update(0.3) == pytest.approx(10.0)

def test_rolling_fps_rejects_invalid_window_size() -> None:
    with pytest.raises(ValueError):
        RollingFps(window_size=1)