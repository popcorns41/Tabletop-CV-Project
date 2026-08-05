from collections import deque
from time import monotonic

class RollingFps:
    """Estimates FPS over a rolling window of frame timestamps."""

    def __init__(self, window_size: int = 30) -> None:
        if window_size < 2:
            raise ValueError("window_size must be at least_2.")

        self._timestamps: deque[float] = deque(maxlen=window_size)


    #This reports the rate at which our application is receiving
    #and processing frames. More useful than blindly trusting CAP_PROP_FPS
    #which some cameras report inaccurately.
    def update(self, timestamp: float | None = None) -> float:
        current_time = monotonic() if timestamp is None else timestamp
        self._timestamps.append(current_time)

        if len(self._timestamps) < 2:
            return 0.0

        elapsed = self._timestamps[-1] - self._timestamps[0]

        if elapsed <= 0.0:
            return 0.0

        intervals = len(self._timestamps) -1
        return intervals / elapsed