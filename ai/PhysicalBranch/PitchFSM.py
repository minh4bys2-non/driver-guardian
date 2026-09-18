import cv2
import mediapipe as mp
import numpy as np
import math
import time
from collections import deque

# --- CẤU HÌNH CÁC ĐIỂM MỐC (LANDMARKS) THEO CHUẨN MEDIAPIPE FACE MESH ---
LANDMARK_INDICES = [1, 152, 263, 33, 291, 61]

MODEL_POINTS_3D = np.array([
    (0.0, 0.0, 0.0),             # Nose tip
    (0.0, 330.0, -65.0),         # Chin
    (225.0, -170.0, -135.0),     # Left eye outer corner
    (-225.0, -170.0, -135.0),    # Right eye outer corner
    (150.0, 150.0, -125.0),      # Left mouth corner
    (-150.0, 150.0, -125.0)      # Right mouth corner
], dtype=np.float64)

FACE_BOUNDING_BOX_3D = np.array([
    [-200, -220, -200], [ 200, -220, -200], [ 200,  300, -200], [-200,  300, -200],
    [-200, -220,  200], [ 200, -220,  200], [ 200,  300,  200], [-200,  300,  200]
], dtype=np.float64)

# --- CẤU HÌNH BỘ LỌC TÍCH LŨY THỜI GIAN (FSM) CHO GÓC PITCH ---
PITCH_DOWN_THRESHOLD = 14.0   # Ngưỡng góc cúi đầu coi là bất thường (độ)
NOD_MIN_DURATION_MS = 800     # Thời gian cúi tối thiểu để tính là gật gù buồn ngủ (ms)
NOD_MAX_DURATION_MS = 3500    # Quá thời gian này tính là ngủ gục sâu hoặc lơ là kéo dài (ms)
WINDOW_SIZE_SEC = 60          # Cửa sổ trượt 1 phút để tính tần suất F_nod
MAX_MISSING_FRAMES = 12       # Số khung hình mất mặt tối đa trước khi reset FSM

def get_camera_matrix(width, height):
    focal_length = width
    center_x = width / 2.0
    center_y = height / 2.0
    return np.array([[focal_length, 0, center_x], [0, focal_length, center_y], [0, 0, 1]], dtype=np.float64)

def rotation_matrix_to_euler_angles(R):
    pitch = math.asin(np.clip(R[2, 1], -1.0, 1.0))
    if abs(R[2, 1]) < 0.9999:
        yaw = math.atan2(-R[2, 0], R[2, 2])
        roll = math.atan2(-R[0, 1], R[1, 1])
    else:
        yaw = math.atan2(R[0, 2], R[0, 0])
        roll = 0.0
    return np.degrees(pitch), np.degrees(yaw), np.degrees(roll)

def estimate_head_pose_solve_pnp(image_points_2d, camera_matrix, dist_coeffs):
    success, rvec, tvec = cv2.solvePnP(MODEL_POINTS_3D, image_points_2d, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    if not success: return None
    R, _ = cv2.Rodrigues(rvec)
    pitch, yaw, roll = rotation_matrix_to_euler_angles(R)
    return pitch, yaw, roll, rvec, tvec

def estimate_head_pose_direct_geometry(landmarks_all, width, height):
    def get_pixel_3d(idx):
        lm = landmarks_all[idx]
        return np.array([lm.x * width, lm.y * height, lm.z * width])
    p_nose = get_pixel_3d(1)
    p_chin = get_pixel_3d(152)
    p_forehead = get_pixel_3d(10)
    p_left_eye = get_pixel_3d(263)
    p_right_eye = get_pixel_3d(33)
    
    v_x = p_left_eye - p_right_eye
    u_x = v_x / np.linalg.norm(v_x)
    v_y = p_chin - p_forehead
    u_y = v_y / np.linalg.norm(v_y)
    u_z = np.cross(u_x, u_y)
    u_z = u_z / np.linalg.norm(u_z)
    u_y = np.cross(u_z, u_x)
    u_y = u_y / np.linalg.norm(u_y)
    
    R = np.column_stack((u_x, u_y, u_z))
    pitch, yaw, roll = rotation_matrix_to_euler_angles(R)
    tvec = p_nose.reshape(3, 1)
    rvec, _ = cv2.Rodrigues(R)
    return pitch, yaw, roll, rvec, tvec

def draw_3d_axes(image, rvec, tvec, camera_matrix, dist_coeffs, origin_point_2d):
    axis_length = 120.0
    axes_3d = np.array([(0.0, 0.0, 0.0), (axis_length, 0.0, 0.0), (0.0, axis_length, 0.0), (0.0, 0.0, axis_length)], dtype=np.float64)
    points_2d, _ = cv2.projectPoints(axes_3d, rvec, tvec, camera_matrix, dist_coeffs)
    points_2d = np.int32(points_2d.reshape(-1, 2))
    cv2.line(image, tuple(points_2d[0]), tuple(points_2d[1]), (0, 0, 255), 3)
    cv2.line(image, tuple(points_2d[0]), tuple(points_2d[2]), (0, 255, 0), 3)
    cv2.line(image, tuple(points_2d[0]), tuple(points_2d[3]), (255, 0, 0), 3)

def draw_3d_bounding_box(image, rvec, tvec, camera_matrix, dist_coeffs):
    points_2d, _ = cv2.projectPoints(FACE_BOUNDING_BOX_3D, rvec, tvec, camera_matrix, dist_coeffs)
    points_2d = np.int32(points_2d.reshape(-1, 2))
    for i, j in [(0,1), (1,2), (2,3), (3,0)]: cv2.line(image, tuple(points_2d[i]), tuple(points_2d[j]), (255, 255, 0), 2)
    for i, j in [(4,5), (5,6), (6,7), (7,4)]: cv2.line(image, tuple(points_2d[i]), tuple(points_2d[j]), (200, 120, 0), 2)
    for i, j in [(0,4), (1,5), (2,6), (3,7)]: cv2.line(image, tuple(points_2d[i]), tuple(points_2d[j]), (0, 255, 255), 2)

def main():
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
        if not cap.isOpened(): return

    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    use_solve_pnp = True
    offset_pitch = offset_yaw = offset_roll = 0.0
    calibrated = False
    prev_time = 0
    missing_face_counter = 0
    
    # --- KHỞI TẠO BIẾN CHO BỘ LỌC TÍCH LŨY THỜI GIAN FSM (PITCH) ---
    head_fsm_state = "NORMAL"         # Trạng thái FSM: "NORMAL" hoặc "PITCH_DOWN"
    pitch_down_start_time = None      # Đánh dấu thời điểm bắt đầu cúi đầu
    nod_timestamps = deque()          # Lưu mốc thời gian các cú gật đầu trong 60s
    last_nod_duration = 0.0           # Thời lượng của cú gật đầu gần nhất (ms)
    head_event_log = "System Stable"  # Nhật ký sự kiện động học của đầu
    nod_frequency = 0                 # Tần suất gật gù F_nod (lần/phút)

    window_name = "Head Pose Estimation 3D - Temporal FSM"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.6, min_tracking_confidence=0.6) as face_mesh:
        while cap.isOpened():
            success, frame = cap.read()
            if not success: break
                
            current_time = time.time()
            fps = 1.0 / (current_time - prev_time) if prev_time > 0 else 0.0
            prev_time = current_time
            
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            camera_matrix = get_camera_matrix(w, h)
            dist_coeffs = np.zeros((4, 1), dtype=np.float64)
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)
            
            pose_status = "No Face"
            
            if results.multi_face_landmarks:
                missing_face_counter = 0
                landmarks = results.multi_face_landmarks[0].landmark
                
                image_points_2d = np.array([[landmarks[idx].x * w, landmarks[idx].y * h] for idx in LANDMARK_INDICES], dtype=np.float64)
                
                for pt in image_points_2d:
                    cv2.circle(frame, (int(pt[0]), int(pt[1])), 4, (0, 255, 255), -1)
                
                mp_drawing.draw_landmarks(
                    image=frame, landmark_list=results.multi_face_landmarks[0],
                    connections=mp_face_mesh.FACEMESH_TESSELATION, landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )
                
                pose_data = estimate_head_pose_solve_pnp(image_points_2d, camera_matrix, dist_coeffs) if use_solve_pnp else estimate_head_pose_direct_geometry(landmarks, w, h)
                    
                if pose_data is not None:
                    raw_pitch, raw_yaw, raw_roll, rvec, tvec = pose_data
                    pitch = raw_pitch - offset_pitch
                    yaw = raw_yaw - offset_yaw
                    roll = raw_roll - offset_roll
                    
                    nose_2d = (int(image_points_2d[0][0]), int(image_points_2d[0][1]))
                    draw_3d_axes(frame, rvec, tvec, camera_matrix, dist_coeffs, nose_2d)
                    draw_3d_bounding_box(frame, rvec, tvec, camera_matrix, dist_coeffs)
                    
                    # ----------------- ĐỘNG LỰC HỌC TÍCH LŨY THỜI GIAN (PITCH FSM) -----------------
                    if pitch > PITCH_DOWN_THRESHOLD:  # Tài xế đang cúi đầu vượt ngưỡng an toàn
                        if head_fsm_state == "NORMAL":
                            head_fsm_state = "PITCH_DOWN"
                            pitch_down_start_time = current_time
                        else:
                            # Đang nằm trong trạng thái cúi đầu, đo đạc thời gian tích lũy
                            current_duration = (current_time - pitch_down_start_time) * 1000
                            last_nod_duration = current_duration
                            
                            if current_duration > NOD_MAX_DURATION_MS:
                                head_event_log = "CRITICAL: PROLONGED MICRO-SLEEP / DISTRACTION"
                            elif current_duration >= NOD_MIN_DURATION_MS:
                                head_event_log = "WARNING: DROWSY HEAD DROPPING DETECTED"
                    else:  # Góc Pitch bình thường (tài xế nhìn thẳng hoặc đã ngẩng đầu lên)
                        if head_fsm_state == "PITCH_DOWN":
                            # Tính toán thời lượng tích lũy của toàn bộ chu kỳ cúi đầu vừa diễn ra
                            duration_ms = (current_time - pitch_down_start_time) * 1000
                            last_nod_duration = duration_ms
                            head_fsm_state = "NORMAL"
                            
                            if NOD_MIN_DURATION_MS <= duration_ms <= NOD_MAX_DURATION_MS:
                                head_event_log = f"F_nod: DROWSY HEAD NOD DETECTED ({duration_ms:.0f}ms)"
                                nod_timestamps.append(current_time)  # Tích lũy sự kiện vào hàng đợi
                            elif duration_ms < NOD_MIN_DURATION_MS:
                                head_event_log = f"INTENTIONAL GLANCE / TAPLO LOOK ({duration_ms:.0f}ms)"
                            pitch_down_start_time = None
                    
                    # Cập nhật cửa sổ trượt 60 giây để tính tần suất F_nod
                    while nod_timestamps and (current_time - nod_timestamps[0] > WINDOW_SIZE_SEC):
                        nod_timestamps.popleft()
                    nod_frequency = len(nod_timestamps)
                    
                    # ----------------- PHÂN LOẠI TRẠNG THÁI & HIỂN THỊ HUD -----------------
                    pose_status = "Looking Center"
                    status_color = (0, 255, 0)
                    
                    if "CRITICAL" in head_event_log or nod_frequency >= 3:
                        pose_status = "CRITICAL: FATIGUE / SLEEP GUCK!"
                        status_color = (0, 0, 255)
                    elif "WARNING" in head_event_log or nod_frequency >= 1:
                        pose_status = "WARNING: DROWSY SIGNALS"
                        status_color = (0, 165, 255)
                    elif yaw > 15: pose_status = "Looking Right"
                    elif yaw < -15: pose_status = "Looking Left"
                    
                    # Mở rộng kích thước hộp HUD để chứa thêm thông số tích lũy FSM
                    hud_overlay = frame.copy()
                    cv2.rectangle(hud_overlay, (15, 15), (380, 290), (30, 30, 30), -1)
                    cv2.addWeighted(hud_overlay, 0.7, frame, 0.3, 0, frame)
                    
                    method_str = "OpenCV SolvePnP" if use_solve_pnp else "Direct 3D Geometry"
                    cv2.putText(frame, f"Method: {method_str}", (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
                    cv2.putText(frame, f"Pitch (Up/Down): {pitch:.2f} deg", (25, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(frame, f"Yaw (Left/Right): {yaw:.2f} deg", (25, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(frame, f"Roll (Tilt L/R):  {roll:.2f} deg", (25, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    # Thêm các thông số tích lũy thời gian động học lên màn hình hiển thị
                    cv2.putText(frame, f"F_nod (Nod Freq): {nod_frequency} nods/min", (25, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 255, 200), 2, cv2.LINE_AA)
                    cv2.putText(frame, f"Z_nod (Duration): {last_nod_duration:.0f} ms", (25, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(frame, f"Log: {head_event_log}", (25, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 150, 255), 1, cv2.LINE_AA)
                    
                    calib_str = "[Calibrated]" if calibrated else "[Press R to Calibrate]"
                    cv2.putText(frame, pose_status, (25, 255), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2, cv2.LINE_AA)
                    cv2.putText(frame, calib_str, (25, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
            else:
                missing_face_counter += 1
                if missing_face_counter > MAX_MISSING_FRAMES:
                    # Reset lại máy trạng thái hữu hạn để tránh kẹt log khi mất dấu khuôn mặt
                    head_fsm_state = "NORMAL"
                    pitch_down_start_time = None
                cv2.putText(frame, "No face detected!", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            
            cv2.putText(frame, f"FPS: {fps:.1f}", (w - 120, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow(window_name, frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('m'):
                use_solve_pnp = not use_solve_pnp
                calibrated = False
                offset_pitch = offset_yaw = offset_roll = 0.0
            elif key == ord('r'):
                if results.multi_face_landmarks and pose_data is not None:
                    offset_pitch, offset_yaw, offset_roll = raw_pitch, raw_yaw, raw_roll
                    calibrated = True
                    
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()