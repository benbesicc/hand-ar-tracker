"""rolling average fps counter for real-time performance monitoring

provides a frame rate calculator that maintains a sliding window
of frame timestamps to compute smooth accurate fps readings
with support for min max and average statistics
"""

import time
from collections import deque
from typing import Optional

class FPSCounter:
    """rolling window fps counter with statistics tracking

    maintains a deque of frame timestamps and computes the rolling
    average frames per second over a configurable window size

    attributes
        windowsize number of frames to include in the rolling average
        timestamps deque of frame arrival timestamps
        minfps lowest fps value observed since last reset
        maxfps highest fps value observed since last reset
    """

    def __init__(self, window_size: int = 30) -> None:
        """initialize the fps counter with a given window size

        args
            windowsize number of recent frames to use for calculating
                the rolling average fps larger values produce smoother
                readings but respond more slowly to changes
        """

        self.window_size: int = max(1, window_size)
        self.timestamps: deque[float] = deque(maxlen=self.window_size)
        self.min_fps: float = float("inf")
        self.max_fps: float = 0.0
        self._frame_count: int = 0

    def tick(self) -> None:
        """record a new frame timestamp

        should be called once per frame at the point where you want
        to measure the frame rate updates internal statistics
        """

        current_time = time.perf_counter()
        self.timestamps.append(current_time)
        self._frame_count += 1
        current_fps = self.get_fps()
        if current_fps > 0.0 and self._frame_count > 1:
            self._update_statistics(current_fps)

    def _update_statistics(self, current_fps: float) -> None:
        """update min and max fps statistics

        args
            currentfps the current fps reading to compare against
                stored min and max values
        """

        if current_fps < self.min_fps:
            self.min_fps = current_fps
        if current_fps > self.max_fps:
            self.max_fps = current_fps

    def get_fps(self) -> float:
        """calculate the current rolling average fps

        computes the average frame rate over the timestamps stored
        in the rolling window

        returns
            the current fps as a float returns 00 if fewer than
            two frames have been recorded
        """

        if len(self.timestamps) < 2:
            return 0.0
        time_span = self.timestamps[-1] - self.timestamps[0]
        if time_span <= 0.0:
            return 0.0
        frame_count = len(self.timestamps) - 1
        return frame_count / time_span

    def get_fps_string(self) -> str:
        """get a formatted string representation of the current fps

        returns
            a string like fps 305 with one decimal place precision
        """

        return f"FPS: {self.get_fps():.1f}"

    def get_min_fps(self) -> float:
        """get the minimum fps observed since initialization or last reset

        returns
            the lowest fps value recorded or 00 if no frames
            have been processed yet
        """

        if self.min_fps == float("inf"):
            return 0.0
        return self.min_fps

    def get_max_fps(self) -> float:
        """get the maximum fps observed since initialization or last reset

        returns
            the highest fps value recorded since initialization
            or last reset
        """

        return self.max_fps

    def get_statistics_string(self) -> str:
        """get a formatted string with fps statistics

        returns
            a string containing current min and max fps values
            formatted for display overlay
        """

        current = self.get_fps()
        minimum = self.get_min_fps()
        maximum = self.get_max_fps()
        return (
            f"FPS: {current:.1f} | "
            f"Min: {minimum:.1f} | "
            f"Max: {maximum:.1f}"
        )

    def get_frame_count(self) -> int:
        """get the total number of frames recorded

        returns
            the total frame count since initialization or last reset
        """

        return self._frame_count

    def reset(self) -> None:
        """reset the fps counter clearing all stored data

        clears the timestamp buffer and resets all statistics
        to their initial values
        """

        self.timestamps.clear()
        self.min_fps = float("inf")
        self.max_fps = 0.0
        self._frame_count = 0
