"""Camera thật → đặc trưng physical_v1 → CNN–ConvGRU → phân loại.

Chỉnh cấu hình bên dưới rồi chạy: python -m ai.OptimalAlgorithsm.inference
Một lần chạy neural phục vụ cả ba phép fusion của TrainingSystem.
"""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import sys
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import numpy as np

from ai.OptimalAlgorithsm.DataSystem.config import DEFAULT_CONFIG
from ai.OptimalAlgorithsm.DataSystem.engine import Engine
from ai.OptimalAlgorithsm.DataSystem.loadvideo2handle import balance_brightness
from ai.OptimalAlgorithsm.DataSystem.neuralnetwork import letterbox
from ai.OptimalAlgorithsm.DataSystem.normalization import FEATURE_ENCODING
from ai.OptimalAlgorithsm.TrainingSystem.fitnessFunction import FEATURES, Fitness
from ai.PhysicalBranch.camera_metrics import CameraMetrics


@dataclass(frozen=True)
class CameraConfig:
    camera: int | str = 0
    checkpoint: Path = Path(__file__).resolve().parents[1] / "LSTM/checkpoints/head_landmarkv1/best.pt"
    device: str | None = None  # None: tự chọn CUDA/CPU.
    interval: float = 1.0
    duration: float | None = None
    output: str | None = None
    no_display: bool = False
    primary_fusion: str = "cooccurrence"  # Nhãn chính; vẫn xuất cả ba chế độ.


CAMERA_CONFIG = CameraConfig()
# Thứ tự: blink_frequency, blink_duration, perclos, yawn_frequency,
# nod_duration, nod_frequency, dominant_head_motion_frequency, neuralnetwork_score, threshold.
# Trọng số thủ công CHƯA TỐI ƯU; w >= 0.5 chọn feature, mặc định dùng cả 8.
FUSION_CANDIDATES = {
    "weighted_mean": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5],
    "noisy_or": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5],
    "cooccurrence": [0.5003395080566406,
                     0.8887009024620056,
                     0.7320020794868469,
                     0.2563990652561188,
                     0.0,
                     0.6491293907165527,
                     0.4759885370731354,
                     0.0,
                     0.07674604654312134],
}
# Gán đường dẫn evaluation.json physical_v1 để thay trọng số thủ công của từng chế độ.
FUSION_REPORTS = {"weighted_mean": None, "noisy_or": None, "cooccurrence": None}


def load_fusion(path, normalizer):
    report = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    schema = report.get("input_schema", {})
    if (schema.get("features") != FEATURES
            or schema.get("feature_encoding") != FEATURE_ENCODING
            or schema.get("normalization_scales") != normalizer.scales):
        raise ValueError("Fusion schema không khớp physical_v1 hiện tại; cần train lại và xuất evaluation.json")
    selected = report["selected_model"]
    if selected["model"] == "candidate":
        Fitness(np.zeros((1, len(FEATURES)))).predict(
            selected["candidate"], report["fitness_method"]
        )
    elif type(selected.get("constant_label")) is not int or selected["constant_label"] not in (0, 1):
        raise ValueError("Fusion model không hợp lệ")
    return report


def classify(sample, fusion):
    selected = fusion["selected_model"]
    if selected["model"] != "candidate":
        return selected["constant_label"]
    return int(Fitness([[sample[name] for name in FEATURES]]).predict(
        selected["candidate"], fusion["fitness_method"]
    )[0])


def fusion_details(sample, fusion):
    selected = fusion["selected_model"]
    score = threshold = None
    if selected["model"] == "candidate":
        candidate = selected["candidate"]
        score = float(Fitness([[sample[name] for name in FEATURES]]).score(candidate, fusion["fitness_method"])[0])
        threshold = float(candidate[8])
    return {"score": score, "threshold": threshold, "prediction": classify(sample, fusion)}


def draw_dashboard(frame, metrics, result, fusions, status, timestamp, fps, args):
    width = 600
    preview = cv2.resize(frame, (width, max(1, round(frame.shape[0] * width / frame.shape[1]))))
    height = max(740, preview.shape[0] + 300)
    canvas = np.full((height, width + 660, 3), 24, dtype=np.uint8)
    canvas[:preview.shape[0], :width] = preview
    white, yellow, green, red = (225, 225, 225), (0, 220, 255), (100, 230, 100), (80, 80, 255)

    def line(text, x, y, color=white):
        cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .49, color, 1, cv2.LINE_AA)

    def number(value, digits=3):
        return "--" if value is None else f"{value:.{digits}f}"

    y = preview.shape[0] + 26
    line(f"LIVE PHYSICAL | {fps:.1f} FPS | {timestamp:.1f}s", 12, y, yellow)
    live = (
        ("EAR / MAR", f"{number(metrics['ear'])} / {number(metrics['mar'])}"),
        ("PERCLOS (%)", number(metrics['perclos_pct'])),
        ("Blink / yawn / nod (per min)", " / ".join(number(metrics[k], 1) for k in
                                                    ('blink_rate_per_min', 'yawning_frequency_per_min',
                                                     'nodding_frequency_per_min'))),
        ("Pitch / yaw / roll (deg)", " / ".join(number(metrics[k], 1) for k in ('pitch_deg', 'yaw_deg', 'roll_deg'))),
        ("Head motion (Hz)", number(metrics['head_motion_frequency_hz'])),
        ("Coverage eye / mouth / head (s)", " / ".join(number(metrics[k], 1) for k in
                                                       ('eye_observed_sec', 'mouth_observed_sec',
                                                        'head_observed_sec'))),
    )
    for label, value in live:
        y += 26
        line(f"{label}: {value}", 12, y)
    y += 26
    ready = "/".join("READY" if metrics[k] else "WAIT" for k in ('eye_ready', 'mouth_ready', 'head_calibrated'))
    line(f"Eye / mouth / head: {ready}", 12, y, yellow)
    line("Face detected" if metrics['face_detected'] else "NO FACE", 12, y + 26,
         green if metrics['face_detected'] else yellow)
    line("Q / ESC: quit", 12, height - 14)

    x = width + 18
    age = timestamp - result['end_second'] if result else None
    fresh = age is not None and age <= args.interval * 2 and metrics['face_detected']
    label = status if result is None or fresh else "STALE - waiting for valid window"
    line(f"{args.primary_fusion}: {label}", x, 28, yellow)
    line("Latest valid window | normalized features [0,1]", x, 58, yellow)
    for i, name in enumerate(FEATURES):
        line(f"{name}: {number(result[name] if result else None, 4)}", x, 86 + i * 25)
    line("Physical features are measurements, not probabilities.", x, 296)
    line("NEURAL: CNN + ConvGRU", x, 332, yellow)
    if result:
        score = result['neuralnetwork_score']
        line(f"P(alert)={1 - score:.4f}   P(drowsy)={score:.4f}   age={age:.1f}s", x, 358)
    else:
        line("Waiting for a valid physical window and neural clip...", x, 358)
    line("FUSION: score / threshold -> prediction", x, 398, yellow)
    for i, (method, fusion) in enumerate(fusions.items()):
        y = 428 + i * 77
        detail = result['fusion_details'][method] if result else None
        threshold = fusion['selected_model'].get('candidate', [None] * 9)[8]
        text = "WAIT" if detail is None else "DROWSY" if detail['prediction'] else "ALERT"
        if detail and not fresh:
            text += " (STALE)"
        color = yellow if not fresh else red if detail['prediction'] else green
        line(method, x, y, yellow)
        line(f"{number(detail['score'] if detail else None, 4)} / {number(threshold, 4)} -> {text}", x, y + 24, color)
        source = "manual, unoptimized" if fusion['source'] == 'manual_unoptimized' else "trained report"
        if fusion['selected_model']['model'] != 'candidate':
            source += " | constant baseline (no score)"
        line(source, x, y + 46)
    line("Fusion scores are not calibrated probabilities.", x, 678)
    return canvas


def load_fusions(normalizer, candidates=None, reports=None):
    candidates = FUSION_CANDIDATES if candidates is None else candidates
    reports = FUSION_REPORTS if reports is None else reports
    fusions = {}
    for method, candidate in candidates.items():
        path = reports.get(method)
        if path is not None:
            fusion = load_fusion(path, normalizer)
            if fusion["fitness_method"] != method:
                raise ValueError(f"Fusion report không đúng chế độ {method}: {path}")
        else:
            Fitness(np.zeros((1, len(FEATURES)))).predict(candidate, method)
            fusion = {"fitness_method": method,
                      "selected_model": {"model": "candidate", "candidate": list(candidate)}}
        fusion["source"] = str(path) if path is not None else "manual_unoptimized"
        fusions[method] = fusion
    return fusions


class NeuralBuffer:
    """Lấy ảnh quá khứ gần nhất theo lưới neural_fps; bỏ clip khi camera gián đoạn."""

    def __init__(self, config):
        self.config = config
        self.frames = deque(maxlen=config.neural_count)
        self.previous = None
        self.previous_time = None
        self.next_time = 0.0

    def update(self, rgb, timestamp):
        if self.previous_time is None or timestamp - self.previous_time > self.config.max_gap_sec:
            self.frames.clear()
            self.next_time = timestamp
        frame = np.moveaxis(letterbox(rgb, self.config.neural_size), -1, 0)
        while self.next_time <= timestamp + 1e-9:
            past = self.previous if self.next_time < timestamp - 1e-9 else frame
            self.frames.append(past)
            self.next_time += 1 / self.config.neural_fps
        self.previous, self.previous_time = frame, timestamp


def run(args=CAMERA_CONFIG):
    if not np.isfinite(args.interval) or args.interval <= 0:
        raise ValueError("interval phải hữu hạn và > 0")
    if args.duration is not None and (not np.isfinite(args.duration) or args.duration <= 0):
        raise ValueError("duration phải hữu hạn và > 0")
    # Giữ config trích xuất để hash không đổi chỉ vì chu kỳ hiển thị trực tiếp.
    config = replace(DEFAULT_CONFIG, checkpoint_path=str(args.checkpoint), device=args.device)
    engine = Engine(Path(__file__).parent, config=config)
    fusions = load_fusions(engine.normalizer)
    mode = args.primary_fusion
    if mode not in fusions:
        raise ValueError(f"primary_fusion không có trong FUSION_CANDIDATES: {mode}")
    for method, fusion in fusions.items():
        print(f"[fusion] {method}: {fusion['source']}, {fusion['selected_model']}", file=sys.stderr)
    print(f"[init] mode={mode}, device={engine.neural.device}, window={config.window_sec:g}s", file=sys.stderr)
    print("[init] Giữ mặt trong khung, nhìn thẳng khi hiệu chuẩn; Q/ESC để thoát.", file=sys.stderr)
    source = int(args.camera) if isinstance(args.camera, str) and args.camera.isdecimal() else args.camera
    capture = cv2.VideoCapture(source)
    camera = CameraMetrics(config=config)
    worker = ThreadPoolExecutor(max_workers=1)
    pending = output = None
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Không mở được camera: {source}")
        capture.set(cv2.CAP_PROP_FPS, config.fps)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if args.output:
            output = Path(args.output).expanduser().open("a", encoding="utf-8")
        buffer = NeuralBuffer(config)
        blink, nod = deque(), deque()
        origin = time.perf_counter()
        last_timestamp = None
        next_prediction = config.window_sec
        result = None
        status = "WARMING UP"
        frames = 0
        while True:
            ok, frame = capture.read()
            timestamp = time.perf_counter() - origin
            if last_timestamp is not None and timestamp <= last_timestamp:
                timestamp = last_timestamp + 1e-4
            last_timestamp = timestamp
            if not ok:
                raise RuntimeError("Không đọc được frame từ camera")
            frame = balance_brightness(frame, config.target_luma)
            metrics = camera.process(frame, timestamp)
            buffer.update(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), timestamp)
            for history, detected, duration in (
                    (blink, "blink_detected", "blink_duration_ms"),
                    (nod, "nod_detected", "nod_event_duration_ms"),
            ):
                if metrics[detected] and metrics[duration] is not None:
                    history.append((timestamp, metrics[duration]))
                while history and history[0][0] <= timestamp - config.window_sec:
                    history.popleft()

            if pending is not None and pending.done():
                result = pending.result()
                pending = None
                status = "WAITING FOR VALID WINDOW"
                if result is not None:
                    result.pop("label")
                    result["source_camera"] = result.pop("source_video")
                    result["fusion_details"] = {
                        method: fusion_details(result, fusion) for method, fusion in fusions.items()
                    }
                    result["fusion_predictions"] = {
                        method: detail["prediction"] for method, detail in result["fusion_details"].items()
                    }
                    result["fusion_sources"] = {method: fusion["source"] for method, fusion in fusions.items()}
                    result["prediction"] = result["fusion_predictions"][mode]
                    result["mode"] = mode
                    line = json.dumps(result, ensure_ascii=False, allow_nan=False)
                    print(line, flush=True)
                    if output:
                        output.write(line + "\n")
                        output.flush()
                    status = "DROWSY" if result["prediction"] else "ALERT"

            if timestamp >= next_prediction and pending is None:
                next_prediction = timestamp + args.interval
                if len(buffer.frames) == config.neural_count:
                    pending = worker.submit(
                        engine._features, metrics.copy(), tuple(blink), tuple(nod),
                        tuple(buffer.frames), str(source), timestamp - config.window_sec, None,
                    )
                else:
                    result = None
                    status = "WAITING FOR CAMERA CLIP"

            frames += 1
            if not args.no_display:
                canvas = draw_dashboard(frame, metrics, result, fusions, status, timestamp,
                                        frames / max(timestamp, 1e-6), args)
                cv2.imshow("Driver Guardian", canvas)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
                if cv2.getWindowProperty("Driver Guardian", cv2.WND_PROP_VISIBLE) < 1:
                    break
            if args.duration is not None and timestamp >= args.duration:
                break
    finally:
        capture.release()
        camera.close()
        worker.shutdown(wait=True, cancel_futures=True)
        if output:
            output.close()
        if not args.no_display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        pass
    except (OSError, ValueError, KeyError, RuntimeError, cv2.error) as exc:
        sys.exit(f"Lỗi: {exc}")
