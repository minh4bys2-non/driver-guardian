import cv2
import torch
import time

# Giả sử bạn đã import NMSFreeInference và NMSFreeDetector từ module của bạn
from src.model import NMSFreeDetector
from src.runtime.infer import NMSFreeInference

def run_camera_detection(detector: NMSFreeInference, camera_id: int = 0):
    # Khởi tạo kết nối Camera (0 thường là Webcam mặc định của máy)
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print(f"Không thể mở Camera ID: {camera_id}")
        return

    print("Đang chạy luồng Camera... Nhấn 'q' để thoát.")
    
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Không thể nhận khung hình từ Camera.")
            break

        # 1. Truyền TRỰC TIẾP `frame` (np.ndarray) vào detector
        result = detector(frame)

        # 2. Lấy kết quả bbox, scores, class_ids
        boxes = result["boxes_xyxy_orig"]
        scores = result["scores"]
        class_ids = result["class_ids"]

        if isinstance(boxes, torch.Tensor): boxes = boxes.cpu().numpy()
        if isinstance(scores, torch.Tensor): scores = scores.cpu().numpy()
        if isinstance(class_ids, torch.Tensor): class_ids = class_ids.cpu().numpy()

        # 3. Vẽ Bounding Box trực tiếp bằng OpenCV (Cực nhanh)
        for box, score, cls in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = map(int, box)
            
            cls_idx = int(cls)
            class_name = detector.class_names[cls_idx] if cls_idx < len(detector.class_names) else str(cls_idx)
            label = f"{class_name} {score:.2f}"

            # Vẽ khung chữ nhật (Màu Xanh Chanh: RGB(0, 255, 0))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Vẽ background cho text nhãn
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - h - 6), (x1 + w, y1), (0, 255, 0), -1)
            
            # Vẽ text
            cv2.putText(frame, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        # 4. Tính toán và hiển thị FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # 5. Hiển thị lên màn hình
        cv2.imshow("NMS-Free Object Detection (Real-time)", frame)

        # Nhấn phím 'q' để dừng
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Khởi tạo mô hình
    model = NMSFreeDetector()
    detector = NMSFreeInference(
        model=model,
        categories_path="/home/tranmanhduy/Workspace/ptithcm/TTTN/CNNModel/checkpoints/categories.jsonl",
        checkpoint_path="/home/tranmanhduy/Workspace/ptithcm/TTTN/CNNModel/checkpoints/best.pt",
        img_size=480,
        device="cuda",
        score_thres=0.5,
        use_nms=False
    )

    # Chạy trực tiếp với Webcam
    run_camera_detection(detector, camera_id=0)