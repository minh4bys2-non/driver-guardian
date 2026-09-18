from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
from src.model import NMSFreeDetector
from src.train.dataloader1_obj365 import letterbox, load_categories
from src.utils.checkpoint import load_model_only

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class NMSFreeInference:
    def __init__(
        self,
        model: torch.nn.Module,
        checkpoint_path: str,
        categories_path: Union[str, Path],
        img_size: int = 480,
        device: Optional[Union[str, torch.device]] = None,
        score_thres: float = 0.25,
        iou_thres: float = 0.45,
        use_nms: bool = False,
        max_det: int = 300,
    ) -> None:
        self.img_size = img_size
        self.score_thres = score_thres
        self.iou_thres = iou_thres
        self.use_nms = use_nms
        self.max_det = max_det

        self.model = model
        load_model_only(model=self.model, map_location="cpu", path=checkpoint_path)
        self.device = torch.device(device) if device is not None else next(model.parameters()).device
        self.model = model.eval().to(self.device)

        cat_id_to_idx, records, _ = load_categories(str(categories_path))
        self.class_names: List[str] = [r.get("name", str(r.get("id", i))) for i, r in enumerate(records)]
        self.contiguous_to_cat_id: Dict[int, int] = {idx: cid for cid, idx in cat_id_to_idx.items()}

    def preprocess(self, image_input: Union[str, Path, np.ndarray]) -> Tuple[np.ndarray, torch.Tensor, float, int, int]:
        if isinstance(image_input, (str, Path)):
            img_bgr = cv2.imread(str(image_input))
            if img_bgr is None:
                raise FileNotFoundError(f"Could not load image: {image_input}")
        else:
            img_bgr = np.ascontiguousarray(image_input.astype(np.uint8))

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_letterboxed, scale, pad_left, pad_top = letterbox(img_rgb, self.img_size)
        tensor = torch.from_numpy(img_letterboxed.transpose(2, 0, 1)).float() / 255.0
        return img_bgr, tensor, scale, pad_left, pad_top

    def _filter(self, cls_logits: torch.Tensor, boxes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scores_all = torch.sigmoid(cls_logits)
        scores, class_ids = scores_all.max(dim=-1)
        keep = scores >= self.score_thres
        scores, class_ids, boxes = scores[keep], class_ids[keep], boxes[keep]

        if boxes.numel() and self.use_nms:
            keep = torchvision.ops.batched_nms(boxes, scores, class_ids, self.iou_thres)
            scores, class_ids, boxes = scores[keep], class_ids[keep], boxes[keep]

        if len(scores) > self.max_det:
            topk = scores.argsort(descending=True)[: self.max_det]
            scores, class_ids, boxes = scores[topk], class_ids[topk], boxes[topk]

        return boxes, scores, class_ids

    def _boxes_to_original(
        self, boxes: torch.Tensor, scale: float, pad_left: int, pad_top: int, orig_shape: Tuple[int, int]
    ) -> torch.Tensor:
        if boxes.numel() == 0:
            return boxes
        h, w = orig_shape
        boxes_orig = boxes.clone()
        boxes_orig[:, [0, 2]] = (boxes[:, [0, 2]] - pad_left) / scale
        boxes_orig[:, [1, 3]] = (boxes[:, [1, 3]] - pad_top) / scale
        boxes_orig[:, [0, 2]].clamp_(0, w)
        boxes_orig[:, [1, 3]].clamp_(0, h)
        return boxes_orig

    def build_batch(
        self, image_inputs: List[Union[str, Path, np.ndarray]]
    ) -> Tuple[List[torch.Tensor], List[np.ndarray], List[float], List[int], List[int]]:
        imgs_bgr, tensors, scales, pads_left, pads_top = [], [], [], [], []
        for img in image_inputs:
            img_bgr, tensor, scale, pad_left, pad_top = self.preprocess(img)
            imgs_bgr.append(img_bgr)
            tensors.append(tensor)
            scales.append(scale)
            pads_left.append(pad_left)
            pads_top.append(pad_top)
        return tensors, imgs_bgr, scales, pads_left, pads_top

    @torch.inference_mode()
    def __call__(
        self, image_input: Union[str, Path, np.ndarray, List[Union[str, Path, np.ndarray]]]
    ) -> Union[Dict, List[Dict]]:
        is_batch = isinstance(image_input, (list, tuple))
        image_list = list(image_input) if is_batch else [image_input]

        tensors, imgs_bgr, scales, pads_left, pads_top = self.build_batch(image_list)
        input_tensor = torch.stack(tensors, dim=0).to(self.device)
        raw_output = self.model(input_tensor)

        results = []
        for i in range(len(image_list)):
            boxes, scores, class_ids = self._filter(raw_output["o2o"]["cls"][i], raw_output["o2o"]["box"][i])
            boxes_orig = self._boxes_to_original(boxes, scales[i], pads_left[i], pads_top[i], imgs_bgr[i].shape[:2])
            results.append({
                "raw_output": raw_output,
                "input_tensor": input_tensor[i : i + 1],
                "orig_image": imgs_bgr[i],
                "boxes_xyxy_orig": boxes_orig,
                "scores": scores,
                "class_ids": class_ids,
            })

        return results if is_batch else results[0]

def visualize_and_save(result: Dict, class_names: List[str], output_dir: str = "./",
                        filename: str = "result.jpg", figsize=(10, 10), dpi=200):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image = result["orig_image"][:, :, ::-1]

    boxes = result["boxes_xyxy_orig"]
    scores = result["scores"]
    class_ids = result["class_ids"]

    if isinstance(boxes, torch.Tensor): boxes = boxes.cpu().numpy()
    if isinstance(scores, torch.Tensor): scores = scores.cpu().numpy()
    if isinstance(class_ids, torch.Tensor): class_ids = class_ids.cpu().numpy()

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(image)
    ax.axis("off")
    for box, score, cls in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = box
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1,
                                linewidth=2, edgecolor="lime", facecolor="none"))
        label = f"{class_names[int(cls)]} {score:.2f}" if cls < len(class_names) else f"{int(cls)} {score:.2f}"
        ax.text(x1, y1, label, fontsize=8, color="white",
                bbox=dict(facecolor="green", alpha=0.8, pad=2, edgecolor="none"))
    save_path = output_dir / filename
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.show()
    plt.close(fig)
    return save_path

if __name__ == "__main__":
    model = NMSFreeDetector.from_config("/home/tranmanhduy/Workspace/ptithcm/TTTN/ObjectDetection_24M/checkpoints_ftCOCO/model_mainfest.json")
    detector = NMSFreeInference(
        model=model,
        categories_path="/home/tranmanhduy/Workspace/ptithcm/TTTN/ObjectDetection_24M/checkpoints_ftCOCO/categories.jsonl",
        img_size=480,
        device="cuda",
        checkpoint_path="/home/tranmanhduy/Workspace/ptithcm/TTTN/ObjectDetection_24M/checkpoints_ftCOCO/ft_step00091000.pt",
        iou_thres=0.45,
        use_nms=False,
        score_thres=0.25,
        max_det=30,
    )

    results = detector(image_input=[
        "/home/tranmanhduy/Workspace/ptithcm/TTTN/ObjectDetection_24M/src/runtime/image.jpg",
        "/home/tranmanhduy/Workspace/ptithcm/TTTN/ObjectDetection_24M/src/runtime/image.png",
        "/home/tranmanhduy/Workspace/ptithcm/TTTN/ObjectDetection_24M/src/runtime/image3.jpg",
    ])
    for idx, res in enumerate(results):
        visualize_and_save(result=res, class_names=detector.class_names,
                            output_dir="/home/tranmanhduy/Workspace/ptithcm/TTTN/CNNModel/src/runtime/result",
                            filename=f"result_{idx}.jpg")