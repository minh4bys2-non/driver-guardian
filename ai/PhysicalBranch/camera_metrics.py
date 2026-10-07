import json
import time
from contextlib import ExitStack

import cv2
import numpy as np

from ai.PhysicalBranch.adaptive_hmm_fsm import AdaptiveHMM_FSM
from ai.PhysicalBranch.head_pose_estimation import HeadPoseEstimator
from ai.PhysicalBranch.temporal_metrics import HeadMotionWindow, StateMachine, WindowRatio, positive_seconds


class FaceTracker:
    def __init__(self, detection_confidence=0.6, tracking_confidence=0.6):
        import mediapipe as mp
        self.mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence)

    def process(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self.mesh.process(rgb)
        if not result.multi_face_landmarks:
            return None
        height, width = frame.shape[:2]
        return np.array([(p.x * width, p.y * height)
                         for p in result.multi_face_landmarks[0].landmark])

    def close(self):
        self.mesh.close()


class TensorBoardLogger:
    def __init__(self, logdir):
        from tensorboard.summary.writer.event_file_writer import EventFileWriter
        self.writer = EventFileWriter(logdir)

    def write(self, metrics, step):
        from tensorboard.compat.proto.event_pb2 import Event
        from tensorboard.compat.proto.summary_pb2 import Summary
        values = [Summary.Value(tag=key, simple_value=float(value))
                  for key, value in metrics.items()
                  if isinstance(value, (bool, int, float, np.number)) and np.isfinite(value)]
        self.writer.add_event(Event(wall_time=time.time(), step=step, summary=Summary(value=values)))

    def close(self):
        self.writer.close()


class CameraMetrics:
    LEFT_EYE = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE = [33, 160, 158, 133, 153, 144]
    MOUTH = [61, 37, 267, 291, 314, 84]

    def __init__(self, fps=30, window_sec=60, calibration_frames=30,
                 eye_options=None, mouth_options=None, angle_limits=(20, 25, 20),
                 nod_pitch_deg=14, nod_release_deg=8, nod_direction=None,
                 nod_duration_ms=(800, 3500), max_gap_sec=0.25,
                 detection_confidence=0.6, tracking_confidence=0.6,
                 windows=None, head_motion_min_amplitude_deg=0.1):
        if calibration_frames < 1 or nod_direction not in (None, -1, 1):
            raise ValueError("Invalid calibration or nod direction")
        if not np.isfinite([nod_release_deg, nod_pitch_deg]).all() or not 0 <= nod_release_deg < nod_pitch_deg:
            raise ValueError("Nod release must be below the pitch threshold")
        self.angle_limits = np.asarray(angle_limits, dtype=float)
        if self.angle_limits.shape != (3,) or not np.isfinite(self.angle_limits).all() or (self.angle_limits <= 0).any():
            raise ValueError("Angle limits must contain three positive finite values")
        common = dict(fps=fps, window_size_sec=window_sec, max_gap_sec=max_gap_sec)
        self.eye = AdaptiveHMM_FSM("eye", **(common | (eye_options or {})))
        self.mouth = AdaptiveHMM_FSM("mouth", **(common | (mouth_options or {})))
        self.nod = StateMachine("pitch", min_duration_ms=nod_duration_ms[0],
                                max_duration_ms=nod_duration_ms[1], **common)
        self.over_angle_ratio = WindowRatio(window_sec, max_gap_sec)
        self.head_motion = HeadMotionWindow(window_sec, max_gap_sec, head_motion_min_amplitude_deg)
        self._windows = dict(blink=self.eye.fsm, perclos=self.eye.ratio,
                             yawn=self.mouth.fsm, pom=self.mouth.ratio,
                             nod=self.nod, over_angle=self.over_angle_ratio, head_motion=self.head_motion)
        self.window_sec = positive_seconds(window_sec)
        self.set_windows(**(windows or {}))
        self.calibration_frames = calibration_frames
        self.nod_pitch, self.nod_release, self.nod_direction = nod_pitch_deg, nod_release_deg, nod_direction
        self.confidence = detection_confidence, tracking_confidence
        self.tracker = None
        self.estimator = None
        self.shape = None
        self.reset()

    def reset(self):
        for detector in (self.eye, self.mouth):
            detector.reset()
        self._reset_head()
        self.timestamp = None

    def _reset_head(self):
        self.nod.reset()
        self.over_angle_ratio.reset()
        self.head_motion.reset()
        self.neutral_samples = []
        self.neutral = None
        self.nodding = False

    @property
    def windows_sec(self):
        return {name: metric.window for name, metric in self._windows.items()}

    def set_windows(self, window_sec=None, **windows):
        """Đổi cửa sổ; xóa lịch sử chỉ số bị đổi, giữ HMM và baseline đầu."""
        unknown = windows.keys() - self._windows.keys()
        if unknown:
            raise ValueError(f"Unknown windows: {sorted(unknown)}")
        configured = self.windows_sec
        if window_sec is not None:
            configured = dict.fromkeys(configured, positive_seconds(window_sec))
        configured.update({name: positive_seconds(value) for name, value in windows.items()})
        for name, value in configured.items():
            metric = self._windows[name]
            if metric.window != value:
                metric.window = value
                metric.reset()
                if name == "nod":
                    self.nodding = False
        if window_sec is not None:
            self.window_sec = float(window_sec)

    @staticmethod
    def aspect_ratio(points, indices):
        p = np.asarray(points, dtype=float)[indices, :2]
        width = np.linalg.norm(p[0] - p[3])
        if not np.isfinite(p).all() or width < 1e-6:
            return None
        return float((np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])) / (2 * width))

    def process_landmarks(self, points, image_size, timestamp):
        timestamp = float(timestamp)
        if not np.isfinite(timestamp) or (self.timestamp is not None and timestamp <= self.timestamp):
            raise ValueError("Timestamps must be finite and strictly increasing")
        if self.timestamp is not None and timestamp - self.timestamp > self.nod.max_gap:
            self.nodding = False
        self.timestamp = timestamp
        if self.shape != tuple(image_size):
            self.estimator = HeadPoseEstimator(*image_size)
            self.shape = tuple(image_size)
            self._reset_head()
        if points is not None:
            points = np.asarray(points, dtype=float)
            if points.ndim != 2 or points.shape[0] < 468 or points.shape[1] != 2 or not np.isfinite(points).all():
                points = None
        ear = mar = angles = raw = None
        pose_error = None
        if points is not None:
            left = self.aspect_ratio(points, self.LEFT_EYE)
            right = self.aspect_ratio(points, self.RIGHT_EYE)
            ear = (left + right) / 2 if left is not None and right is not None else None
            mar = self.aspect_ratio(points, self.MOUTH)
            try:
                raw = np.asarray(self.estimator.estimate(points[self.estimator.LANDMARKS]))
                if self.neutral is None:
                    self.neutral_samples.append(raw)
                    if len(self.neutral_samples) >= self.calibration_frames:
                        self.neutral = np.median(self.neutral_samples, axis=0)
                        self.neutral_samples.clear()
                if self.neutral is not None:
                    angles = (raw - self.neutral + 180) % 360 - 180
            except (ValueError, cv2.error) as exc:
                pose_error = str(exc)
                self.neutral_samples.clear()
        elif self.neutral is None:
            self.neutral_samples.clear()
        eye = self.eye.process(ear, timestamp)
        mouth = self.mouth.process(mar, timestamp)
        over_angle = nod_state = None
        if angles is not None:
            over_angle = bool(np.any(np.abs(angles) > self.angle_limits))
            pitch_down = abs(angles[0]) if self.nod_direction is None else self.nod_direction * angles[0]
            self.nodding = bool(pitch_down > (self.nod_release if self.nodding else self.nod_pitch))
            nod_state = int(self.nodding)
        else:
            self.nodding = False
        nod = self.nod.process(nod_state, timestamp)
        over_percent, head_seconds = self.over_angle_ratio.process(over_angle, timestamp)
        motion = self.head_motion.process(None if angles is None else angles[0], timestamp)
        return {
            "timestamp_sec": timestamp,
            "window_sec": self.window_sec,
            "windows_sec": self.windows_sec,
            "face_detected": points is not None,
            "eye_ready": eye["initialized"],
            "mouth_ready": mouth["initialized"],
            "head_ready": angles is not None,
            "head_calibrated": self.neutral is not None,
            "pose_valid": raw is not None,
            "pose_error": pose_error,
            "eye_state": eye["state"],
            "mouth_state": mouth["state"],
            "ear": ear,
            "perclos_pct": eye["perclos"],
            "p80_ear_threshold": eye["p80_threshold"],
            "blink_rate_per_min": eye["rate_per_minute"],
            "blink_detected": eye["event_done"],
            "blink_count": eye["event_count"],
            "blink_duration_ms": eye["last_valid_duration_ms"],
            "eye_closure_duration_ms": eye["current_duration_ms"] if ear is not None and eye["initialized"] else None,
            "last_eye_closure_duration_ms": eye["last_event_duration_ms"],
            "eye_observed_sec": eye["observed_seconds"],
            "mar": mar,
            "pom_pct": mouth["pom"],
            "yawning_frequency_per_min": mouth["rate_per_minute"],
            "yawn_detected": mouth["event_done"],
            "yawn_count": mouth["event_count"],
            "yawn_duration_ms": mouth["last_valid_duration_ms"],
            "mouth_open_duration_ms": mouth["current_duration_ms"] if mar is not None and mouth["initialized"] else None,
            "last_mouth_open_duration_ms": mouth["last_event_duration_ms"],
            "mouth_observed_sec": mouth["observed_seconds"],
            "pitch_deg": None if angles is None else float(angles[0]),
            "pitch_amplitude_deg": None if angles is None else float(abs(angles[0])),
            "yaw_deg": None if angles is None else float(angles[1]),
            "roll_deg": None if angles is None else float(angles[2]),
            "over_angle": over_angle,
            "over_angle_pct": over_percent,
            "nodding": None if nod_state is None else bool(nod_state),
            "nod_detected": nod["event_done"],
            "nod_count": nod["event_count"],
            "nodding_frequency_per_min": nod["rate_per_minute"],
            "nod_duration_ms": nod["current_duration_ms"] if nod_state is not None else None,
            "last_nod_duration_ms": nod["last_event_duration_ms"],
            "nod_event_duration_ms": nod["last_valid_duration_ms"],
            "head_observed_sec": head_seconds,
            **motion,
        }

    def process(self, frame, timestamp=None):
        if self.tracker is None:
            self.tracker = FaceTracker(*self.confidence)
        timestamp = time.monotonic() if timestamp is None else timestamp
        return self.process_landmarks(self.tracker.process(frame), frame.shape[1::-1], timestamp)

    @staticmethod
    def draw(frame, metrics):
        groups = (
            ("EYES", "eye", "ear", (
                "ear", "eye_state", "perclos_pct", "blink_rate_per_min", "blink_count", "blink_duration_ms",
                "eye_closure_duration_ms", "last_eye_closure_duration_ms",
                "p80_ear_threshold", "eye_observed_sec")),
            ("MOUTH", "mouth", "mar", (
                "mar", "mouth_state", "pom_pct", "yawning_frequency_per_min", "yawn_count", "yawn_duration_ms",
                "mouth_open_duration_ms", "last_mouth_open_duration_ms", "mouth_observed_sec")),
            ("HEAD", "head", "pitch_deg", (
                "pitch_deg", "yaw_deg", "roll_deg", "pitch_amplitude_deg", "over_angle_pct",
                "nodding", "nod_detected", "nodding_frequency_per_min",
                "nod_count", "nod_duration_ms", "nod_event_duration_ms", "head_observed_sec",
                "pitch_mean_deg", "pitch_mean_amplitude_deg", "head_motion_frequency_hz",
                "head_motion_resolution_hz", "head_motion_observed_sec")),
        )
        height = max(frame.shape[0], 110 + 22 * sum(len(keys) + 1 for _, _, _, keys in groups))
        width = frame.shape[1]
        canvas = np.full((height, width + 460, 3), (24, 24, 24), dtype=np.uint8)
        canvas[:frame.shape[0], :width] = frame
        x, y = width + 16, 28

        def line(label, value=None, color=(225, 225, 225)):
            nonlocal y
            if isinstance(value, (bool, np.bool_)):
                value = "YES" if value else "NO"
            elif isinstance(value, (int, float, np.number)):
                value = f"{value:.2f}"
            text = label if value is None else f"{label}: {value}"
            cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 1, cv2.LINE_AA)
            y += 22

        for title, kind, signal, keys in groups:
            status = "READY" if metrics[f"{kind}_ready"] else "CALIBRATING"
            if not metrics["face_detected"]:
                status = "NO FACE"
            elif kind == "head" and metrics["pose_error"]:
                status = "POSE LOST"
            elif kind != "head" and metrics[signal] is None:
                status = "NO SIGNAL"
            y += 6
            line(title, status, color=(80, 210, 255))
            for key in keys:
                value = metrics[key]
                if key in ("eye_state", "mouth_state"):
                    states = ("OPEN", "CLOSED") if key == "eye_state" else ("RESTING", "OPEN")
                    value = None if value is None else states[value]
                line(key.replace("_", " "), "--" if value is None else value)
        line("FPS / processing", f"{metrics.get('fps', 0):.1f} / {metrics.get('processing_ms', 0):.1f} ms")
        ready = all(metrics[key] for key in ('eye_ready', 'mouth_ready', 'head_ready'))
        status = 'Angles relative to calibrated neutral pose' if ready else 'Calibration: look straight, eyes open, mouth relaxed'
        if not metrics['face_detected']:
            status = 'NO FACE: waiting for valid landmarks'
        elif metrics['pose_error']:
            status = metrics['pose_error']
        cv2.putText(canvas, status[:78], (12, height - 36), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (0, 220, 255), 1)
        cv2.putText(canvas, '[r] reset calibration and history    [q] quit', (12, height - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.43, (220, 220, 220), 1)
        return canvas

    def close(self):
        if self.tracker is not None:
            self.tracker.close()
            self.tracker = None

    def run(self, source=0, display=True, logdir=None, output_interval_sec=1,
            max_frames=None, use_video_time=False, on_metrics=None):
        if output_interval_sec < 0 or (max_frames is not None and max_frames < 1):
            raise ValueError("Invalid output interval or frame limit")
        self.reset()
        with ExitStack() as stack:
            stack.callback(self.close)
            cap = cv2.VideoCapture(source)
            stack.callback(cap.release)
            if not cap.isOpened():
                raise RuntimeError(f'Cannot open camera/video: {source}')
            fps = cap.get(cv2.CAP_PROP_FPS)
            if use_video_time and (not np.isfinite(fps) or fps <= 0):
                raise ValueError('Video must have valid FPS to use its timeline')
            logger = TensorBoardLogger(logdir) if logdir else None
            if logger:
                stack.callback(logger.close)
            if display:
                stack.callback(cv2.destroyAllWindows)
            start = previous = time.monotonic()
            next_output = 0.0
            step = 0
            while max_frames is None or step < max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                tick = time.monotonic()
                timestamp = step / fps if use_video_time else tick - start
                metrics = self.process(frame, timestamp)
                metrics['processing_ms'] = (time.monotonic() - tick) * 1000
                metrics['fps'] = 1 / max(tick - previous, 1e-6)
                if logger:
                    logger.write(metrics, step)
                if on_metrics is not None:
                    on_metrics(metrics)
                if timestamp >= next_output:
                    print(json.dumps(metrics, allow_nan=False), flush=True)
                    next_output = timestamp + output_interval_sec
                if display:
                    if metrics['pose_valid']:
                        self.estimator.draw_axes(frame)
                    cv2.imshow('Camera metrics', self.draw(frame, metrics))
                    key = cv2.waitKey(1) & 0xff
                    if key == ord('q'):
                        break
                    if key == ord('r'):
                        self.reset()
                previous = tick
                step += 1
        return step


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Physical metrics from camera/video")
    parser.add_argument("--source", default="0", help="Camera index or video path")
    parser.add_argument("--window-sec", type=float, default=60)
    parser.add_argument("--metric-window", action="append", default=[], metavar="NAME=SECONDS",
                        help="Override blink/perclos/yawn/pom/nod/over_angle/head_motion; repeatable")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--video-time", action="store_true")
    args = parser.parse_args()
    try:
        windows = {name: float(value) for name, value in
                   (item.split("=", 1) for item in args.metric_window)}
        monitor = CameraMetrics(window_sec=args.window_sec, windows=windows)
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    source = int(args.source) if args.source.isdecimal() else args.source
    monitor.run(source=source, display=not args.no_display, use_video_time=args.video_time)
