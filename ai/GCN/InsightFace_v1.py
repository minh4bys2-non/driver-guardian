import time
import cv2
import numpy as np
import onnxruntime as ort
import os
from insightface.app import FaceAnalysis

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def init_insightface_gpu(det_size=(640, 640)):
    # Kiểm tra xem ONNX Runtime có nhận GPU CUDA không
    available_providers = ort.get_available_providers()
    print(f"[INFO] ONNX Runtime Providers khả dụng: {available_providers}")

    if 'CUDAExecutionProvider' not in available_providers:
        print("[WARNING] Không tìm thấy CUDAExecutionProvider! Mô hình sẽ chạy trên CPU.")

    # Chỉ nạp module 'detection' (SCRFD) và 'landmark_3d_68' (1k3d68.onnx)
    # Bỏ qua recognition và genderage để tăng tối đa FPS
    app = FaceAnalysis(
        name='buffalo_l',
        allowed_modules=['detection', 'landmark_3d_68'],
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )

    # ctx_id=0 chỉ định GPU số 0; det_size=(640, 640) là kích thước ảnh đầu vào cho detector
    app.prepare(ctx_id=0, det_size=det_size)
    return app


def draw_68_landmarks(frame, landmarks_3d, draw_connections=True):
    """
    Vẽ 68 điểm mốc (chuẩn iBUG 68) lên khung hình.
    landmarks_3d có shape (68, 3) gồm tọa độ (x, y, z).
    """
    pts = landmarks_3d[:, :2].astype(np.int32)

    if draw_connections:
        # Định nghĩa các vùng nối điểm theo chuẩn 68 landmarks: (start, end, is_closed, color)
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
        for start, end, is_closed, color in regions:
            cv2.polylines(frame, [pts[start:end]], isClosed=is_closed, color=color, thickness=1, lineType=cv2.LINE_AA)

    # Vẽ từng điểm mốc
    for (x, y) in pts:
        cv2.circle(frame, (x, y), radius=2, color=(0, 255, 0), thickness=-1, lineType=cv2.LINE_AA)


def main():
    # 1. Khởi tạo mô hình trên GPU CUDA
    app = init_insightface_gpu(det_size=(640, 640))

    # 2. Mở kết nối Camera (0 là webcam mặc định, hoặc truyền đường dẫn file video / RTSP)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("[ERROR] Không thể mở camera!")
        return

    print("[INFO] Đang chạy... Nhấn phím 'q' trên cửa sổ video để thoát.")
    prev_time = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 3. Suy luận trên GPU: trả về danh sách các khuôn mặt trong khung hình
        faces = app.get(frame)

        for face in faces:
            # --- A. Vẽ Bounding Box ---
            bbox = face.bbox.astype(np.int32)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)

            # --- B. Lấy và vẽ 68 điểm mốc 3D (x, y, z) ---
            if face.landmark_3d_68 is not None:
                lmk_68 = face.landmark_3d_68  # Shape: (68, 3)
                draw_68_landmarks(frame, lmk_68, draw_connections=True)

            # --- C. Lấy góc nghiêng đầu 3D (Pitch, Yaw, Roll) đi kèm mô hình 1k3d68 ---
            if face.pose is not None:
                pitch, yaw, roll = face.pose
                pose_text = f"Pitch: {pitch:.1f} | Yaw: {yaw:.1f} | Roll: {roll:.1f}"
                cv2.putText(frame, pose_text, (bbox[0], max(20, bbox[1] - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        # 4. Tính toán và hiển thị tốc độ FPS thực tế
        curr_time = time.perf_counter()
        fps = 1.0 / (curr_time - prev_time)
        prev_time = curr_time

        cv2.putText(frame, f"FPS (CUDA): {fps:.1f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        # 5. Hiển thị kết quả
        cv2.imshow("InsightFace 68 3D Landmarks (CUDA)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
