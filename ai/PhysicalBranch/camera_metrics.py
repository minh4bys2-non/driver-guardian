import json
import time
from contextlib import ExitStack

import cv2
import numpy as np

from adaptive_hmm_fsm import AdaptiveHMM_FSM, StateMachine, WindowRatio
from head_pose_estimation import HeadPoseEstimator


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
                 nod_pitch_deg=14, nod_release_deg=8, nod_direction=1,
                 nod_duration_ms=(800, 3500), max_gap_sec=0.25,
                 detection_confidence=0.6, tracking_confidence=0.6):
        if calibration_frames < 1 or nod_direction not in (-1, 1):
            raise ValueError("Invalid calibration or nod direction")
        if not 0 <= nod_release_deg < nod_pitch_deg:
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
        self.calibration_frames = calibration_frames
        self.nod_pitch, self.nod_release, self.nod_direction = nod_pitch_deg, nod_release_deg, nod_direction
        self.confidence = detection_confidence, tracking_confidence
        self.tracker = None
        self.estimator = None
        self.shape = None
        self.reset()

    def reset(self):
        for detector in (self.eye, self.mouth, self.nod, self.over_angle_ratio):
            detector.reset()
        self.neutral_samples = []
        self.neutral = None
        self.nodding = False
        self.timestamp = None

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
            self.neutral_samples.clear()
            self.neutral = None
            self.nodding = False
            self.nod.reset()
            self.over_angle_ratio.reset()
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
            pitch_down = self.nod_direction * angles[0]
            self.nodding = bool(pitch_down > (self.nod_release if self.nodding else self.nod_pitch))
            nod_state = int(self.nodding)
        else:
            self.nodding = False
        nod = self.nod.process(nod_state, timestamp)
        over_percent, head_seconds = self.over_angle_ratio.process(over_angle, timestamp)
        return {
            "timestamp_sec": timestamp,
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
            "eye_closure_duration_ms": eye["current_duration_ms"] if ear is not None and eye["initialized"] else None,
            "last_eye_closure_duration_ms": eye["last_event_duration_ms"],
            "eye_observed_sec": eye["observed_seconds"],
            "mar": mar,
            "pom_pct": mouth["pom"],
            "yawning_frequency_per_min": mouth["rate_per_minute"],
            "yawn_detected": mouth["event_done"],
            "mouth_open_duration_ms": mouth["current_duration_ms"] if mar is not None and mouth["initialized"] else None,
            "last_mouth_open_duration_ms": mouth["last_event_duration_ms"],
            "mouth_observed_sec": mouth["observed_seconds"],
            "pitch_deg": None if angles is None else float(angles[0]),
            "yaw_deg": None if angles is None else float(angles[1]),
            "roll_deg": None if angles is None else float(angles[2]),
            "over_angle": over_angle,
            "over_angle_pct": over_percent,
            "nodding": None if nod_state is None else bool(nod_state),
            "nod_detected": nod["event_done"],
            "nodding_frequency_per_min": nod["rate_per_minute"],
            "nod_duration_ms": nod["current_duration_ms"] if nod_state is not None else None,
            "last_nod_duration_ms": nod["last_event_duration_ms"],
            "head_observed_sec": head_seconds,
        }

    def process(self, frame, timestamp=None):
        if self.tracker is None:
            self.tracker = FaceTracker(*self.confidence)
        timestamp = time.monotonic() if timestamp is None else timestamp
        return self.process_landmarks(self.tracker.process(frame), frame.shape[1::-1], timestamp)

    @staticmethod
    def draw(frame, metrics):
        height = max(frame.shape[0], 700)
        width = frame.shape[1]
        canvas = np.full((height, width + 460, 3), (24, 24, 24), dtype=np.uint8)
        canvas[:frame.shape[0], :width] = frame
        x, y = width + 16, 28

        def line(label, value=None, unit="", color=(225, 225, 225)):
            nonlocal y
            if isinstance(value, (bool, np.bool_)):
                value = "YES" if value else "NO"
            elif isinstance(value, (int, float, np.number)):
                value = f"{value:.2f}"
            text = label if value is None else f"{label}: {value} {unit}"
            cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 1, cv2.LINE_AA)
            y += 22

        def metric(label, key, unit=""):
            value = metrics[key]
            line(label, "--" if value is None else value, unit)

        def section(title, status):
            nonlocal y
            y += 6
            line(title, status, color=(80, 210, 255))

        def signal_status(kind, value_key):
            if metrics[value_key] is None:
                return "NO SIGNAL"
            return "READY" if metrics[f"{kind}_ready"] else "CALIBRATING"

        section("EYES", signal_status("eye", "ear"))
        metric("EAR", "ear")
        line("Eye state", {0: "OPEN", 1: "CLOSED"}.get(metrics['eye_state'], "--"))
        metric("PERCLOS P80", "perclos_pct", "%")
        metric("Blink rate", "blink_rate_per_min", "/min")
        metric("Closure now / last", "eye_closure_duration_ms", f"/ {metrics['last_eye_closure_duration_ms']:.0f} ms")
        metric("P80 EAR threshold", "p80_ear_threshold")
        metric("Observed", "eye_observed_sec", "s")
        section("MOUTH", signal_status("mouth", "mar"))
        metric("MAR", "mar")
        line("Mouth state", {0: "RESTING", 1: "OPEN"}.get(metrics['mouth_state'], "--"))
        metric("POM", "pom_pct", "%")
        metric("Yawn frequency", "yawning_frequency_per_min", "/min")
        metric("Open now / last", "mouth_open_duration_ms", f"/ {metrics['last_mouth_open_duration_ms']:.0f} ms")
        metric("Observed", "mouth_observed_sec", "s")
        head_status = "READY" if metrics['head_ready'] else ("POSE LOST" if metrics['pose_error'] else "CALIBRATING")
        if not metrics['face_detected']:
            head_status = "NO FACE"
        section("HEAD", head_status)
        for label, key in (("Pitch", "pitch_deg"), ("Yaw", "yaw_deg"), ("Roll", "roll_deg")):
            metric(label, key, "deg")
        metric("Over-angle", "over_angle")
        metric("Over-angle time", "over_angle_pct", "%")
        metric("Nodding / nod completed", "nodding", f"/ {metrics['nod_detected']}")
        metric("Nod frequency", "nodding_frequency_per_min", "/min")
        metric("Nod now / last", "nod_duration_ms", f"/ {metrics['last_nod_duration_ms']:.0f} ms")
        metric("Observed", "head_observed_sec", "s")
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
    monitor = CameraMetrics(
        fps=30,
        window_sec=60,
        calibration_frames=30,
        eye_options=dict(init_duration_sec=5, min_duration_ms=100, max_duration_ms=2000,
                         learning_rate=0.01, adapt_interval=300),
        mouth_options=dict(init_duration_sec=5, min_duration_ms=3500, max_duration_ms=7500,
                           learning_rate=0.01, adapt_interval=300),
        angle_limits=(20, 25, 20),
        nod_pitch_deg=14,
        nod_release_deg=8,
        nod_direction=1,
        nod_duration_ms=(800, 3500),
        max_gap_sec=0.25,
        detection_confidence=0.6,
        tracking_confidence=0.6,
    )
    monitor.run(source=0, display=True, logdir=None, output_interval_sec=1)
