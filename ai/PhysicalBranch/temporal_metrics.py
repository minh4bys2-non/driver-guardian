from collections import deque

import numpy as np


def positive_seconds(value):
    if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
        raise ValueError("Time parameters must be finite and positive")
    return float(value)


class StateMachine:

    def __init__(self, mode="eye", fps=30, window_size_sec=60,
                 min_duration_ms=None, max_duration_ms=None, max_gap_sec=0.25):
        limits = {"eye": (100, 2000), "mouth": (3500, 7500), "pitch": (800, 3500)}
        if mode not in limits:
            raise ValueError("Invalid FSM configuration")
        self.mode = mode
        self.fps = positive_seconds(fps)
        self.window = positive_seconds(window_size_sec)
        lo, hi = limits[mode]
        self.minimum = lo if min_duration_ms is None else min_duration_ms
        self.maximum = hi if max_duration_ms is None else max_duration_ms
        if not np.isfinite([self.minimum, self.maximum]).all() or not 0 <= self.minimum < self.maximum:
            raise ValueError("Invalid event duration limits")
        self.max_gap = positive_seconds(max_gap_sec)
        self.reset()

    def reset(self):
        self.time = None
        self.start = None
        self.last_duration = 0.0
        self.last_valid_duration = None
        self.armed = False
        self.events = deque()

    def process(self, state, timestamp=None):
        now = (0 if self.time is None else self.time + 1 / self.fps) if timestamp is None else float(timestamp)
        if not np.isfinite(now) or (self.time is not None and now <= self.time):
            raise ValueError("Timestamps must be finite and strictly increasing")
        if state not in (0, 1, None):
            raise ValueError("State must be 0, 1 or None")
        if self.time is not None and now - self.time > self.max_gap:
            self.start = None
            self.armed = False
        self.time = now
        done = False
        if state is None:
            self.start = None
            self.armed = False
        elif state == 1 and self.start is None:
            self.start = now
        elif state == 0 and self.start is not None:
            self.last_duration = (now - self.start) * 1000
            done = self.armed and self.minimum <= self.last_duration <= self.maximum
            if done:
                self.events.append(now)
                self.last_valid_duration = self.last_duration
            self.start = None
        if state == 0:
            self.armed = True
        while self.events and self.events[0] <= now - self.window:
            self.events.popleft()
        duration = 0 if self.start is None else (now - self.start) * 1000
        rate = len(self.events) * 60 / self.window
        return {
            "state": state, "signal_missing": state is None,
            "event_done": done, "current_duration_ms": duration,
            "last_event_duration_ms": self.last_duration,
            "last_valid_duration_ms": self.last_valid_duration,
            "event_count": len(self.events),
            "rate_per_minute": rate, "prolonged": duration > self.maximum,
        }


class WindowRatio:
    def __init__(self, window_sec=60, max_gap_sec=0.25):
        self.window = positive_seconds(window_sec)
        self.max_gap = positive_seconds(max_gap_sec)
        self.reset()

    def reset(self):
        self.intervals = deque()
        self.previous = None
        self.time = None

    def process(self, active, timestamp):
        if not np.isfinite(timestamp) or (self.time is not None and timestamp <= self.time):
            raise ValueError("Timestamps must be finite and strictly increasing")
        self.time = timestamp
        if self.previous is not None and active is not None:
            start, previous_active = self.previous
            if timestamp - start <= self.max_gap:
                self.intervals.append((start, timestamp, previous_active))
        self.previous = None if active is None else (timestamp, bool(active))
        cutoff = timestamp - self.window
        while self.intervals and self.intervals[0][1] <= cutoff:
            self.intervals.popleft()
        valid = total = 0.0
        for start, end, state in self.intervals:
            duration = end - max(start, cutoff)
            valid += duration
            total += duration * state
        return (100 * total / valid if valid else None), valid


class HeadMotionWindow:
    """Thống kê pitch tương đối; FFT chỉ dùng cửa sổ đầy, liên tục."""

    def __init__(self, window_sec=20, max_gap_sec=0.25, min_amplitude_deg=0.1):
        self.window = positive_seconds(window_sec)
        self.max_gap = positive_seconds(max_gap_sec)
        if not np.isfinite(min_amplitude_deg) or min_amplitude_deg < 0:
            raise ValueError("Minimum motion amplitude must be finite and nonnegative")
        self.min_amplitude = min_amplitude_deg
        self.reset()

    def reset(self):
        self.samples = deque()
        self.time = None

    def process(self, pitch, timestamp):
        if not np.isfinite(timestamp) or (self.time is not None and timestamp <= self.time):
            raise ValueError("Timestamps must be finite and strictly increasing")
        if self.time is not None and timestamp - self.time > self.max_gap:
            self.samples.clear()
        self.time = timestamp
        if pitch is None or not np.isfinite(pitch):
            self.samples.clear()
        else:
            self.samples.append((timestamp, float(pitch)))
        cutoff = timestamp - self.window
        while len(self.samples) > 1 and self.samples[1][0] <= cutoff:
            self.samples.popleft()
        result = dict(pitch_mean_deg=None, pitch_mean_amplitude_deg=None,
                      head_motion_frequency_hz=None, head_motion_resolution_hz=None,
                      head_motion_observed_sec=0.0)
        if len(self.samples) < 2:
            return result
        times, values = np.asarray(self.samples).T
        start = max(times[0], cutoff)
        observed = timestamp - start
        clipped_times = np.r_[start, times[times > start]]
        clipped_values = np.interp(clipped_times, times, values)
        durations = np.diff(clipped_times)
        left, right = clipped_values[:-1], clipped_values[1:]
        amplitudes = (np.abs(left) + np.abs(right)) / 2
        crossing = (left < 0) & (right > 0) | (left > 0) & (right < 0)
        # Tích phân |pitch| phải tách hai tam giác khi đoạn nội suy đi qua 0.
        amplitudes[crossing] = (left[crossing] ** 2 + right[crossing] ** 2) / (
            2 * (np.abs(left[crossing]) + np.abs(right[crossing])))
        result.update(
            pitch_mean_deg=float(np.sum((left + right) * durations / 2) / observed),
            pitch_mean_amplitude_deg=float(np.sum(amplitudes * durations) / observed),
            head_motion_observed_sec=float(observed),
        )
        if observed < self.window - 1e-9 or len(times) < 4:
            return result
        # Nội suy theo timestamp để FPS dao động không làm đổi đơn vị tần số.
        count = max(4, round(observed / np.median(np.diff(times))) + 1)
        grid = np.linspace(start, timestamp, count)
        signal = np.interp(grid, times, values)
        dt = observed / (count - 1)
        result["head_motion_resolution_hz"] = 1 / (count * dt)
        if np.ptp(signal) <= self.min_amplitude:
            return result
        spectrum = np.abs(np.fft.rfft((signal - signal.mean()) * np.hanning(count))) ** 2
        peak = int(np.argmax(spectrum[1:])) + 1
        result["head_motion_frequency_hz"] = float(np.fft.rfftfreq(count, dt)[peak])
        return result
