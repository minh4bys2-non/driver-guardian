import json
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

class ONNXDetector:
    def __init__(self, backbone_neck_path, head_path, categories_path,
                 score_thres=0.25, max_det=300, providers=None):
        if not 0 <= score_thres <= 1 or max_det <= 0:
            raise ValueError("score_thres phải trong [0, 1], max_det phải > 0")
        categories = sorted(json.loads(Path(categories_path).read_text(encoding="utf-8")),
                            key=lambda c: c["index"])
        if not categories or [c["index"] for c in categories] != list(range(len(categories))):
            raise ValueError("Categories phải có index liên tục từ 0")
        if len({c["id"] for c in categories}) != len(categories):
            raise ValueError("Categories có id trùng nhau")
        self.class_names = [c["name"] for c in categories]
        self.category_ids = np.array([c["id"] for c in categories], dtype=np.int64)
        providers = providers or ["CPUExecutionProvider"]
        self.backbone_neck = ort.InferenceSession(str(backbone_neck_path), providers=providers)
        self.head = ort.InferenceSession(str(head_path), providers=providers)
        image_input = self.backbone_neck.get_inputs()[0]
        if image_input.name != "images" or image_input.shape[1:] != [3, 640, 640] or image_input.type != "tensor(float)":
            raise ValueError("Model phải nhận images float32 [batch_size, 3, 640, 640]")
        features = {v.name: (v.shape, v.type) for v in self.backbone_neck.get_outputs()}
        if features != {v.name: (v.shape, v.type) for v in self.head.get_inputs()}:
            raise ValueError("Đầu ra backbone_neck không khớp đầu vào head")
        outputs = {v.name: v.shape for v in self.head.get_outputs()}
        if outputs["logits"][-1] != len(categories):
            raise ValueError("Số categories không khớp số lớp của head")
        self.score_thres, self.max_det = score_thres, max_det

    @staticmethod
    def preprocess(image):
        if isinstance(image, (str, Path)):
            path = image
            image = cv2.imread(str(path))
            if image is None:
                raise FileNotFoundError(f"Không thể đọc ảnh: {path}")
        if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or not image.size:
            raise ValueError("Ảnh đầu vào phải là BGR uint8 H×W×3")
        h, w = image.shape[:2]
        scale = min(640 / h, 640 / w)
        new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
        left, top = (640 - new_w) // 2, (640 - new_h) // 2
        canvas = np.full((640, 640, 3), 114, dtype=np.uint8)
        canvas[top:top + new_h, left:left + new_w] = cv2.resize(
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        tensor = np.ascontiguousarray(canvas.transpose(2, 0, 1), dtype=np.float32) / 255.0
        return image, tensor, scale, left, top

    def __call__(self, image_input):
        is_batch = isinstance(image_input, (list, tuple))
        images = image_input if is_batch else [image_input]
        if not images:
            return []
        batch = [self.preprocess(image) for image in images]
        inputs = np.stack([item[1] for item in batch])
        features = self.backbone_neck.run(["p3", "p4", "p5"], {"images": inputs})
        logits, predictions = self.head.run(["logits", "boxes"], dict(zip(("p3", "p4", "p5"), features)))
        results = []
        for i, (image, _, scale, left, top) in enumerate(batch):
            scores = np.exp(-np.logaddexp(0, -logits[i])).ravel()
            indices = np.flatnonzero(np.isfinite(scores) & (scores >= self.score_thres))
            if indices.size > self.max_det:
                indices = indices[np.argpartition(scores[indices], -self.max_det)[-self.max_det:]]
            indices = indices[np.argsort(-scores[indices])]
            classes = indices % len(self.class_names)
            boxes = predictions[i, indices // len(self.class_names)].copy()
            boxes[:, 0::2] = np.clip((boxes[:, 0::2] - left) / scale, 0, image.shape[1])
            boxes[:, 1::2] = np.clip((boxes[:, 1::2] - top) / scale, 0, image.shape[0])
            keep = np.isfinite(boxes).all(1) & (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            results.append({"orig_image": image, "boxes_xyxy_orig": boxes[keep],
                            "scores": scores[indices][keep], "class_ids": classes[keep],
                            "category_ids": self.category_ids[classes[keep]]})
        return results if is_batch else results[0]

    def draw(self, result):
        image = result["orig_image"].copy()
        for box, score, cls in zip(result["boxes_xyxy_orig"], result["scores"], result["class_ids"]):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"{self.class_names[cls]} {score:.2f}", (x1, max(15, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        return image


def run_camera(detector, camera_id):
    cap = cv2.VideoCapture(camera_id)
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Không thể mở camera {camera_id}")
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Không thể đọc frame từ camera")
            start = time.perf_counter()
            frame = detector.draw(detector(frame))
            fps = 1 / max(time.perf_counter() - start, 1e-6)
            cv2.putText(frame, f"FPS: {fps:.1f}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("ONNX Detection", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def benchmark(detector, batch_size=1, warmup=5, runs=50):
    if batch_size < 1 or warmup < 0 or runs < 1:
        raise ValueError("batch_size và runs phải > 0, warmup phải >= 0")
    inputs = np.zeros((batch_size, 3, 640, 640), dtype=np.float32)
    timings = []
    for i in range(warmup + runs):
        start = time.perf_counter()
        features = detector.backbone_neck.run(["p3", "p4", "p5"], {"images": inputs})
        middle = time.perf_counter()
        detector.head.run(["logits", "boxes"], dict(zip(("p3", "p4", "p5"), features)))
        end = time.perf_counter()
        if i >= warmup:
            timings.append((middle - start, end - middle, end - start))
    timings = np.array(timings) * 1000
    print(f"ONNX benchmark: batch={batch_size}, warmup={warmup}, runs={runs}")
    print(f"Providers: {detector.backbone_neck.get_providers()}")
    for name, values in zip(("Backbone + neck", "Head", "Total"), timings.T):
        print(f"{name}: mean={values.mean():.2f} ms, p50={np.median(values):.2f} ms, "
              f"p95={np.percentile(values, 95):.2f} ms")
    print(f"Throughput: {batch_size * 1000 / timings[:, 2].mean():.2f} images/s "
          "(ONNX only, không gồm preprocess/postprocess)")

def main():
    backbone_neck_path = "/home/tranmanhduy/Workspace/ptithcm/TTTN/NewVersionObDetect/runtime/backbone_neck.onnx"
    head_path = "/home/tranmanhduy/Workspace/ptithcm/TTTN/NewVersionObDetect/runtime/head.onnx"
    categories_path = "/home/tranmanhduy/Workspace/ptithcm/TTTN/NewVersionObDetect/runtime/categories.json"
    images = ["/home/tranmanhduy/Workspace/ptithcm/TTTN/NewVersionObDetect/inference/image.jpg"]
    camera_id = None  # Camera: đặt images = [] và camera_id = 0.
    score_thres = 0.3
    max_det = 300
    providers = ["CPUExecutionProvider"]
    run_benchmark = True
    benchmark_batch_size = 5
    benchmark_warmup = 5
    benchmark_runs = 50
    output_dir = Path("/home/tranmanhduy/Workspace/ptithcm/TTTN/NewVersionObDetect/runtime/results")

    if bool(images) == (camera_id is not None):
        raise ValueError("Chọn một nguồn: images hoặc camera_id")
    detector = ONNXDetector(backbone_neck_path, head_path, categories_path, score_thres, max_det, providers)
    if run_benchmark:
        benchmark(detector, benchmark_batch_size, benchmark_warmup, benchmark_runs)
    if camera_id is not None:
        run_camera(detector, camera_id)
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, (path, result) in enumerate(zip(images, detector(images))):
        output = output_dir / f"{i:04d}_{Path(path).stem}.jpg"
        if not cv2.imwrite(str(output), detector.draw(result)):
            raise OSError(f"Không thể lưu ảnh: {output}")
        print(f"{path}: {len(result['scores'])} detections -> {output}")

if __name__ == "__main__":
    main()
