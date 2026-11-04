import scipy.integrate
if not hasattr(scipy.integrate, "simps"):
    scipy.integrate.simps = scipy.integrate.simpson

import torchlm  # Dòng import cũ của bạn đặt ở sau
import time
import cv2
import numpy as np
import torch
import torchlm
from torchlm.tools import faceboxesv2
from torchlm.models import pipnet


def init_pipnet_68_gpu(backbone="resnet18"):
    """
    Khởi tạo pipeline FaceBoxesV2 + PIPNet (68 điểm chuẩn 300W) trên CUDA.
    backbone hỗ trợ: 'resnet18' (nhanh nhất ~200 FPS), 'resnet50', 'resnet101' (chính xác nhất), 'mobilenet_v2'
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Đang khởi tạo PIPNet ({backbone} - 68 điểm 300W) trên thiết bị: {device.upper()}")

    if device == "cpu":
        print("[WARNING] Không tìm thấy GPU CUDA! Mô hình sẽ chạy trên CPU (~35.7 FPS).")

    # 1. Gắn bộ phát hiện khuôn mặt FaceBoxesV2 lên GPU
    torchlm.runtime.bind(faceboxesv2(device=device))

    # 2. Gắn mô hình PIPNet cấu hình 68 điểm (300W) lên GPU
    # Lưu ý: với tập 300W (68 điểm), bắt buộc đặt num_lms=68 và meanface_type="300w"
    torchlm.runtime.bind(
        pipnet(
            backbone=backbone,
            pretrained=True,
            num_nb=10,
            num_lms=68,
            net_stride=32,
            input_size=256,
            meanface_type="300w",
            map_location=device
        )
    )
    print("[INFO] Đã nạp xong trọng số mô hình!")


def draw_68_landmarks_custom(frame, landmarks, bboxes, conf_threshold=0.6):
    """
    Vẽ Bounding Box và 68 điểm mốc theo từng vùng giải phẫu (chuẩn 300W / iBUG 68).
    - landmarks: mảng numpy shape (N, 68, 2)
    - bboxes: mảng numpy shape (N, 5) gồm [x1, y1, x2, y2, score]
    """
    regions = [
        (0, 17, False, (255, 200, 0)),  # Đường viền hàm (0 - 16)
        (17, 22, False, (0, 255, 255)),  # Lông mày trái (17 - 21)
        (22, 27, False, (0, 255, 255)),  # Lông mày phải (22 - 26)
        (27, 31, False, (255, 0, 255)),  # Sống mũi (27 - 30)
        (31, 36, False, (255, 0, 255)),  # Chóp mũi dưới (31 - 35)
        (36, 42, True, (0, 255, 0)),  # Mắt trái (36 - 41)
        (42, 48, True, (0, 255, 0)),  # Mắt phải (42 - 47)
        (48, 60, True, (0, 165, 255)),  # Viền môi ngoài (48 - 59)
        (60, 68, True, (0, 0, 255))  # Viền môi trong (60 - 67)
    ]

    for i in range(len(bboxes)):
        x1, y1, x2, y2, score = bboxes[i]
        if score < conf_threshold:
            continue

        # 1. Vẽ khung bao khuôn mặt và độ tin cậy
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, f"Face: {score:.2f}", (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

        # 2. Vẽ các đường nối vùng giải phẫu trên mặt
        pts = landmarks[i].astype(np.int32)  # Shape: (68, 2)
        for start, end, is_closed, color in regions:
            cv2.polylines(frame, [pts[start:end]], isClosed=is_closed, color=color, thickness=1, lineType=cv2.LINE_AA)

        # 3. Vẽ 68 điểm mốc
        for (x, y) in pts:
            cv2.circle(frame, (x, y), radius=2, color=(0, 255, 0), thickness=-1, lineType=cv2.LINE_AA)


def main():
    # 1. Khởi tạo PIPNet 68 điểm trên GPU CUDA
    init_pipnet_68_gpu(backbone="resnet18")

    # 2. Mở luồng Camera qua OpenCV (0 là webcam mặc định)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("[ERROR] Không thể mở camera!")
        return

    print("[INFO] Đang chạy suy luận thời gian thực... Nhấn phím 'q' để thoát.")
    prev_time = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 3. Suy luận trực tiếp trên GPU thông qua torchlm runtime
        # Trả về: landmarks có shape (N, 68, 2) và bboxes có shape (N, 5)
        landmarks, bboxes = torchlm.runtime.forward(frame)

        # 4. Vẽ kết quả nếu phát hiện ít nhất 1 khuôn mặt
        if len(bboxes) > 0:
            # Cách 1: Dùng hàm vẽ tùy chỉnh (nối đường viền mắt, môi, hàm rõ ràng)
            draw_68_landmarks_custom(frame, landmarks, bboxes, conf_threshold=0.6)

            # Cách 2 (Tùy chọn): Dùng hàm vẽ tích hợp sẵn cực gọn của torchlm
            # frame = torchlm.utils.draw_bboxes(frame, bboxes=bboxes)
            # frame = torchlm.utils.draw_landmarks(frame, landmarks=landmarks)

        # 5. Tính toán và hiển thị FPS
        curr_time = time.perf_counter()
        fps = 1.0 / (curr_time - prev_time)
        prev_time = curr_time

        cv2.putText(frame, f"PIPNet-68 FPS (CUDA): {fps:.1f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # 6. Hiển thị lên màn hình
        cv2.imshow("PIPNet 68 Landmarks (300W - CUDA)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()