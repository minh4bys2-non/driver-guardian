import time
from pathlib import Path

import cv2
import numpy as np
import torch

from src.model import NMSFreeDetector
from train.dataloader_ import letterbox
from utils.artifacts import validate_metadata


class NMSFreeInference:
    def __init__(self, checkpoint_path, device=None, img_size=None, score_thres=0.25, max_det=300):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        metadata = validate_metadata(checkpoint_path, checkpoint)
        self.img_size = img_size if img_size is not None else checkpoint.get("cfg", {}).get("img_size", 640)
        if self.img_size <= 0 or self.img_size % max(metadata["architecture"]["strides"]):
            raise ValueError("img_size phải dương và chia hết cho stride lớn nhất")
        if not 0 <= score_thres <= 1 or max_det <= 0:
            raise ValueError("score_thres phải trong [0, 1], max_det phải > 0")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = NMSFreeDetector(**metadata["architecture"], img_size=self.img_size)
        self.model.load_state_dict(checkpoint.get("ema") or checkpoint["model"])
        self.model.to(self.device).eval()
        self.class_names = [c["name"] for c in metadata["categories"]]
        self.category_ids = torch.tensor([c["id"] for c in metadata["categories"]])
        self.score_thres, self.max_det = score_thres, max_det

    def preprocess(self, image):
        if isinstance(image, (str, Path)):
            path = image
            image = cv2.imread(str(path))
            if image is None:
                raise FileNotFoundError(f"Không thể đọc ảnh: {path}")
        if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or not image.size:
            raise ValueError("Ảnh đầu vào phải là BGR uint8 với shape H×W×3")
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized, scale, left, top = letterbox(rgb, self.img_size)
        tensor = torch.from_numpy(np.ascontiguousarray(resized.transpose(2, 0, 1))).float() / 255.
        return image, tensor, scale, left, top

    def _filter(self, logits, boxes):
        scores = logits.sigmoid().flatten()
        scores, indices = scores.topk(min(self.max_det, scores.numel()))
        keep = scores >= self.score_thres
        scores, indices = scores[keep], indices[keep]
        return boxes[indices // self.model.nc], scores, indices % self.model.nc

    @torch.inference_mode()
    def __call__(self, image_input):
        is_batch = isinstance(image_input, (list, tuple))
        images = list(image_input) if is_batch else [image_input]
        if not images:
            return []
        batch = [self.preprocess(image) for image in images]
        tensors = torch.stack([item[1] for item in batch]).to(self.device)
        preds = self.model(tensors, o2o_only=True)["o2o"]
        results = []
        for i, (image, _, scale, left, top) in enumerate(batch):
            boxes, scores, classes = self._filter(preds["cls"][i], preds["box"][i])
            boxes, scores, classes = boxes.cpu(), scores.cpu(), classes.cpu()
            boxes[:, 0::2] = ((boxes[:, 0::2] - left) / scale).clamp(0, image.shape[1])
            boxes[:, 1::2] = ((boxes[:, 1::2] - top) / scale).clamp(0, image.shape[0])
            keep = torch.isfinite(boxes).all(1) & (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            results.append({"orig_image": image, "boxes_xyxy_orig": boxes[keep],
                            "scores": scores[keep], "class_ids": classes[keep],
                            "category_ids": self.category_ids[classes[keep]]})
        return results if is_batch else results[0]


def draw_detections(result, class_names):
    image = result["orig_image"].copy()
    for box, score, cls in zip(result["boxes_xyxy_orig"], result["scores"], result["class_ids"]):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"{class_names[int(cls)]} {float(score):.2f}",
                    (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return image


def run_camera_detection(detector, camera_id=0):
    cap = cv2.VideoCapture(camera_id)
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Không thể mở camera {camera_id}")
        while True:
            start = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Không thể đọc khung hình từ camera")
            frame = draw_detections(detector(frame), detector.class_names)
            fps = 1 / max(time.perf_counter() - start, 1e-6)
            cv2.putText(frame, f"FPS: {fps:.1f}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("NMS-Free Detection", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

def main():
    checkpoint_path = "/home/tranmanhduy/Workspace/ptithcm/TTTN/NewVersionObDetect/checkpoints/checkpoints/2bfb6cc36ef5ab822a944006c746e5d348d67e387501c9027df5b957cf124031/d9afe7f332a0080b29fdd068bbb94f32147807d062bc8dbfeb01f8b968f5ad22/train/best.pt"  # File .pt kèm architecture.json và categories.json.
    images = ["/home/tranmanhduy/Workspace/ptithcm/TTTN/NewVersionObDetect/inference/image.jpg"]  # Ví dụ: ["data/test.jpg"]; đặt camera_id=None khi dùng ảnh.
    camera_id = None
    device = "cuda"  # None: tự chọn CUDA/CPU.
    img_size = None  # None: lấy từ checkpoint.
    score_thres = 0.25
    max_det = 300
    output_dir = Path("runs/inference")

    if not checkpoint_path:
        raise ValueError("Hãy đặt checkpoint_path trong main()")
    if bool(images) == (camera_id is not None):
        raise ValueError("Chọn một nguồn: images hoặc camera_id")
    detector = NMSFreeInference(checkpoint_path, device, img_size, score_thres, max_det)
    if camera_id is not None:
        run_camera_detection(detector, camera_id)
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, path in enumerate(images):
        path = Path(path)
        result = detector(path)
        output = output_dir / f"{i:04d}_{path.stem}.jpg"
        if not cv2.imwrite(str(output), draw_detections(result, detector.class_names)):
            raise OSError(f"Không thể lưu ảnh: {output}")
        print(f"{path}: {len(result['scores'])} detections -> {output}")

if __name__ == "__main__":
    main()
