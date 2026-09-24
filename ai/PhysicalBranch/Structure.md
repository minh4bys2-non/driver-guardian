# KIẾN TRÚC VÀ CƠ CHẾ HOẠT ĐỘNG - PHYSICAL BRANCH (NHÁNH PHÂN TÍCH THỂ CHẤT)

Tài liệu này mô tả chi tiết kiến trúc, các thuật toán toán học, cấu trúc mã nguồn và luồng vận hành của thư mục `ai/PhysicalBranch` thuộc hệ thống giám sát và cảnh báo trạng thái tài xế **Driver Guardian**.

---

## 1. TỔNG QUAN HỆ THỐNG (SYSTEM OVERVIEW)

Nhánh **Physical Branch** chịu trách nhiệm thu thập, phân tích và trích xuất các đặc trưng sinh trắc học và động học thời gian thực của tài xế từ camera cabin. Các chỉ số chính bao gồm:
* **Động thái mắt (Ocular Dynamics)**: Tỷ lệ mở mắt (Eye Aspect Ratio - EAR), trạng thái đóng/mở mắt, tần suất chớp mắt (Blink rate), thời gian nhắm mắt liên tục và chỉ số **PERCLOS (P80)** (tỷ lệ thời gian mắt nhắm $\ge 80\%$ trong cửa sổ trượt).
* **Động thái miệng (Oral Dynamics)**: Tỷ lệ mở miệng (Mouth Aspect Ratio - MAR), phát hiện hành vi ngáp (Yawning), thời gian mở miệng liên tục và chỉ số **POM** (Percentage of Open Mouth).
* **Tư thế và động học đầu (Head Pose Dynamics)**: Ước lượng góc xoay 3D (Pitch - cúi/ngửa, Yaw - quay trái/phải, Roll - nghiêng trái/phải), tự động hiệu chuẩn tư thế trung hòa (Neutral Pose Calibration), phát hiện hành vi gật gù buồn ngủ (Head Nodding / Micro-sleep) và tỷ lệ thời gian mất tập trung nhìn lệch hướng (Over-angle Ratio).

### Đặc tính nổi bật của nhánh PhysicalBranch:
1. **Khả năng tự thích ứng trực tuyến (Online Adaptive HMM)**: Sử dụng mô hình Markov ẩn 2 trạng thái tự động cập nhật phân phối tham số theo từng tài xế (mắt một mí/hai mí, đeo kính, góc camera, ánh sáng thay đổi ngày/đêm) mà không dùng ngưỡng cứng (hardcoded threshold).
2. **Lọc nhiễu thời gian động học (Temporal State Machine - FSM)**: Phân biệt chính xác giữa các phản xạ tự nhiên (chớp mắt nhanh 150ms, liếc taplo 400ms) với các dấu hiệu ngủ gật/mất tập trung nguy hiểm (nhắm mắt kéo dài $>2$s, gật đầu $800-3500$ms).
3. **Cửa sổ trượt chuẩn hóa (Sliding Window Ratio)**: Tính toán chính xác theo thời gian thực các chỉ số y khoa tích lũy trong 60 giây (PERCLOS, POM, OverAngle).
4. **Chuẩn hóa giao diện đầu ra (Standardized DTO)**: Xuất kết quả qua `AIResult` định dạng JSON, sẵn sàng tích hợp với mô hình LSTM chuỗi thời gian hoặc ứng dụng Android Automotive (`PhysicalBranch_kotlin`).

---

## 2. CẤU TRÚC THƯ MỤC VÀ TỆP TIN (DIRECTORY STRUCTURE)

```text
ai/PhysicalBranch/
├── interface.py              # Data Transfer Object (AIResult) chuẩn hóa dữ liệu đầu ra
├── head_pose_estimation.py   # Ước lượng tư thế đầu 3D (Pitch, Yaw, Roll) bằng SQPnP + Levenberg-Marquardt
├── adaptive_hmm_fsm.py       # Lõi thuật toán: HMM tự thích ứng, FSM lọc thời gian, WindowRatio tính PERCLOS/POM
├── PitchFSM.py               # Mô-đun thử nghiệm độc lập và HUD phân tích động học gật gù góc Pitch
├── camera_metrics.py         # Bộ điều phối trung tâm (Pipeline Coordinator), FaceTracker, Calibration & HUD
└── Structure.md              # Tài liệu phân tích kiến trúc và cơ chế hoạt động (tệp này)
```

---

## 3. SƠ ĐỒ LUỒNG DỮ LIỆU VÀ KIẾN TRÚC (DATA FLOW & ARCHITECTURE)

```mermaid
flowchart TD
    VideoInput["Khung hình Video / Webcam (BGR Frame)"] --> FaceTracker["FaceTracker (MediaPipe Face Mesh 468+ Landmarks)"]
    
    subgraph Trích xuất đặc trưng hình học (Geometric Feature Extraction)
        FaceTracker --> EyeLM["6 mốc mắt trái (362, 385, 387, 263, 373, 380)<br/>6 mốc mắt phải (33, 160, 158, 133, 153, 144)"]
        FaceTracker --> MouthLM["6 mốc miệng (61, 37, 267, 291, 314, 84)"]
        FaceTracker --> HeadLM["6 mốc chuẩn 3D (1, 152, 263, 33, 291, 61)"]
        
        EyeLM --> EARCalc["Tính EAR (Eye Aspect Ratio) = (Left + Right)/2"]
        MouthLM --> MARCalc["Tính MAR (Mouth Aspect Ratio)"]
        HeadLM --> PnP["HeadPoseEstimator (solvePnP SQPnP + Refine LM)"]
        PnP --> RawAngles["Góc thô: Pitch, Yaw, Roll"]
    end

    subgraph Hiệu chuẩn và chuẩn hóa (Calibration & Normalization)
        RawAngles --> CalibNode{"Hiệu chuẩn 30 frame đầu?"}
        CalibNode -- "Đang hiệu chuẩn" --> MedianBuffer["Thu thập Median Neutral Pose"]
        CalibNode -- "Đã có Neutral Pose" --> AngleNormalize["Bù góc: Delta Theta = (Theta - Neutral + 180) % 360 - 180"]
    end

    subgraph Lõi phân tích thích ứng và máy trạng thái (Adaptive HMM & Temporal FSM)
        EARCalc --> EyeHMM["AdaptiveHMM ('eye', positive_state='low')<br/>EM Online Adaptation (alpha=0.01)"]
        EyeHMM --> EyeFSM["StateMachine ('eye': 100ms - 2000ms)<br/>Blink Detection & Blink Rate"]
        EyeHMM --> EyeRatio["WindowRatio (60s Window)<br/>Tính PERCLOS P80 (%)"]
        
        MARCalc --> MouthHMM["AdaptiveHMM ('mouth', positive_state='high')<br/>EM Online Adaptation (alpha=0.01)"]
        MouthHMM --> MouthFSM["StateMachine ('mouth': 3500ms - 7500ms)<br/>Yawn Detection & Yawn Rate"]
        MouthHMM --> MouthRatio["WindowRatio (60s Window)<br/>Tính POM (%)"]
        
        AngleNormalize --> PitchHysteresis["Cơ chế trễ Pitch (Trigger > 14°, Release < 8°)"]
        PitchHysteresis --> NodFSM["StateMachine ('pitch': 800ms - 3500ms)<br/>Nod Detection & Frequency"]
        AngleNormalize --> AngleLimitCheck["Kiểm tra góc lệch (Limits: 20°, 25°, 20°)"]
        AngleLimitCheck --> OverAngleRatio["WindowRatio (60s Window)<br/>Tính Over-angle Time (%)"]
    end

    subgraph Tổng hợp và xuất kết quả (Aggregation & Output)
        EyeFSM & EyeRatio & MouthFSM & MouthRatio & NodFSM & OverAngleRatio --> MasterCoord["CameraMetrics.process_landmarks()"]
        MasterCoord --> MetricsDict["Dictionary Metrics chi tiết"]
        MetricsDict --> DTO["interface.py (AIResult Dataclass)"]
        DTO --> JSONOut["JSON Output (to_json)"]
        MetricsDict --> HUD["Trực quan hóa OpenCV HUD (CameraMetrics.draw)"]
        MetricsDict --> TBLogger["TensorBoardLogger (Ghi log sự kiện)"]
    end
```

---

## 4. PHÂN TÍCH CHI TIẾT TỪNG TỆP MÃ NGUỒN (DETAILED CODE ANALYSIS)

### 4.1. `interface.py` - Giao diện kết quả chuẩn hóa (AIResult DTO)

#### Mục đích:
Định nghĩa một cấu trúc dữ liệu bất biến (immutable dataclass) làm cầu nối trung gian giữa nhánh Physical Branch với các nhánh khác trong hệ thống (như nhánh học sâu LSTM, mô-đun cảnh báo bằng giọng nói, backend server hoặc UI dashboard).

#### Cấu trúc lớp `AIResult`:
```python
@dataclass(frozen=True)
class AIResult:
    timestamp_sec: float                        # Mốc thời gian đo lường (giây tính từ đầu phiên/video)
    window_sec: float = 60.0                    # Kích thước cửa sổ trượt tính PERCLOS/tần suất (mặc định 60s)

    # Chỉ số mắt
    ear: float | None = None                    # Eye Aspect Ratio tức thời
    blink_rate_per_min: float | None = None     # Tần suất chớp mắt (lần/phút)
    eye_closure_duration_ms: float | None = None# Thời gian nhắm mắt liên tục hiện tại (ms)
    perclos_pct: float | None = None            # Tỷ lệ PERCLOS P80 (0 - 100%)

    # Chỉ số miệng
    mar: float | None = None                    # Mouth Aspect Ratio tức thời
    mouth_open_duration_ms: float | None = None # Thời gian mở miệng liên tục hiện tại (ms - nhận diện ngáp)

    # Chỉ số tư thế đầu
    pitch_deg: float | None = None              # Góc cúi/ngửa đã bù tư thế trung hòa (độ)
    pitch_deviation_duration_ms: float | None = None # Thời gian pitch lệch ngưỡng liên tục (ms)
    pitch_deviation_rate_per_min: float | None = None# Tần suất pitch lệch ngưỡng (lần/phút)

    # Chỉ số tích hợp bậc cao
    lstm_drowsiness_probability: float | None = None # Xác suất buồn ngủ từ mô hình LSTM (0.0 - 1.0)
    warning_score: float | None = None          # Điểm cảnh báo tổng hợp (0.0: an toàn, 1.0: nguy cấp)

    def to_json(self) -> str:
        return json.dumps(asdict(self), allow_nan=False)
```

#### Nguyên lý vận hành:
* Dùng `@dataclass(frozen=True)` nhằm ngăn chặn việc sửa đổi dữ liệu ngoài ý muốn sau khi đã khởi tạo, đảm bảo tính toàn vẹn khi truyền nhận bất đồng bộ qua các luồng xử lý.
* `to_json()` sử dụng cờ `allow_nan=False` để kiểm soát chặt chẽ: bất kỳ giá trị không hợp lệ (`NaN`, `Inf`) nào đều sẽ báo lỗi thay vì tạo ra chuỗi JSON sai chuẩn, bảo đảm dữ liệu đầu ra an toàn tuyệt đối.

---

### 4.2. `head_pose_estimation.py` - Ước lượng tư thế đầu 3D (Head Pose Estimator)

#### Mục đích:
Xác định hướng quay của đầu trong không gian 3 chiều:
* **Pitch (Trục X)**: Cúi đầu xuống (+) hoặc ngửa đầu lên (-).
* **Yaw (Trục Y)**: Quay đầu sang trái (+) hoặc sang phải (-).
* **Roll (Trục Z)**: Nghiêng đầu sang vai trái (+) hoặc vai phải (-).

#### Cấu trúc lớp `HeadPoseEstimator`:
1. **Mô hình 3D khuôn mặt chuẩn (`MODEL`)**: Gồm 6 điểm giải phẫu chuẩn nhân trắc học:
   ```python
   LANDMARKS = [1, 152, 263, 33, 291, 61]
   MODEL = np.array([
       (0, 0, 0),        # 1: Đỉnh chóp mũi (Nose tip)
       (0, 330, 65),     # 152: Điểm cằm (Chin)
       (225, -170, 135), # 263: Khóe mắt ngoài bên trái (Left eye outer corner)
       (-225, -170, 135),# 33: Khóe mắt ngoài bên phải (Right eye outer corner)
       (150, 150, 125),  # 291: Mép môi bên trái (Left mouth corner)
       (-150, 150, 125)  # 61: Mép môi bên phải (Right mouth corner)
   ], dtype=float)
   ```
2. **Ma trận nội thông số Camera (`camera_matrix`)**:
   Nếu không truyền ma trận camera hiệu chuẩn thực tế từ trước, lớp sẽ tự suy biến một ma trận Pinhole Camera xấp xỉ từ độ phân giải ảnh:
   $$K = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix} = \begin{bmatrix} W & 0 & W/2 \\ 0 & W & H/2 \\ 0 & 0 & 1 \end{bmatrix}$$
   với $W$ là chiều rộng và $H$ là chiều cao khung hình.

#### Thuật toán ước lượng (`estimate`):
1. **Kiểm tra tính suy biến (Degeneracy Check)**:
   ```python
   if np.linalg.matrix_rank(points - points.mean(axis=0)) < 2:
       raise ValueError("Degenerate landmarks")
   ```
   Ngăn ngừa lỗi tính toán khi các điểm mốc bị thẳng hàng hoặc dồn cục do lỗi nhận diện khuôn mặt.
2. **Giải bài toán PnP với SQPnP**:
   Sử dụng `cv2.solvePnP` với cờ `SOLVEPNP_SQPNP` (Sequential Quadratic Programming for PnP). Đây là phương pháp giải PnP phi lặp hiện đại, chính xác cao và không bị rơi vào cực tiểu cục bộ như phương pháp lặp DLT/EPnP thông thường đối với tập điểm nhỏ ($N=6$).
3. **Tinh chỉnh phi tuyến Levenberg-Marquardt**:
   Dùng `cv2.solvePnPRefineLM` để tối ưu hóa sai số chiếu ngược (re-projection error):
   $$\min_{R, t} \sum_{i=1}^N \| p_i - \pi(K, R P_i + t) \|^2$$
4. **Kiểm tra mặt sau camera (Cheirality condition)**:
   Kiểm tra $(R \cdot P_i + t)_z > 0$, loại bỏ các nghiệm hình học ảo nằm phía sau camera.
5. **Chuyển đổi sang góc Euler**:
   Sử dụng phân tích RQ trên ma trận quay $R$ (`cv2.RQDecomp3x3`) để trích xuất trực tiếp các góc `(pitch, yaw, roll)` theo độ.
6. **Vẽ trục tọa độ 3D (`draw_axes`)**:
   Dùng `cv2.projectPoints` chiếu 3 trục (X: Đỏ, Y: Xanh lá, Z: Xanh dương) từ chóp mũi ra không gian ảnh để trực quan hóa hướng nhìn.

---

### 4.3. `adaptive_hmm_fsm.py` - Lõi nhận diện thích ứng và máy trạng thái

Tệp này gồm 4 lớp tạo thành kiến trúc phân tầng:

```text
[ Dữ liệu thô EAR / MAR ]
          │
          ▼
   AdaptiveHMM          <--- Tự động phân cụm & học thích ứng trực tuyến
          │ (0 / 1)
          ▼
   StateMachine         <--- Lọc nhiễu thời gian, tính tần suất, đo thời lượng
          │
          ▼
   WindowRatio          <--- Tính tỷ lệ thời gian trượt (PERCLOS, POM, OverAngle)
          │
          ▼
 AdaptiveHMM_FSM        <--- Bộ điều phối phức hợp cấp cao
```

#### A. Lớp `AdaptiveHMM` (Mô hình Markov ẩn tự thích ứng)
* **Bài toán đặt ra**: Ngưỡng nhắm mắt EAR cố định (ví dụ $EAR < 0.2$) thường thất bại với tài xế mắt nhỏ, góc camera chéo, hoặc khi tài xế đeo kính/mệt mỏi cơ mặt. Cần một mô hình tự học phân phối riêng của từng người.
* **Cấu trúc tham số**:
  * Số trạng thái: $N = 2$ (0: Bình thường/Mở, 1: Tích cực/Đóng hoặc Mở lớn).
  * Vector phân phối ban đầu $\pi = [\pi_0, \pi_1]$.
  * Ma trận xác suất chuyển trạng thái $A = \begin{bmatrix} a_{00} & a_{01} \\ a_{10} & a_{11} \end{bmatrix}$.
  * Phân phối phát xạ Gauss 1 chiều: Trạng thái $k$ có kỳ vọng $\mu_k$ và phương sai $\sigma_k^2$.
* **Cơ chế khởi tạo (`fit_initial`)**:
  1. Phân cụm 1 chiều K-Means (`_kmeans_1d`) trên tập dữ liệu hiệu chuẩn 5 giây đầu.
  2. Khởi tạo sơ bộ $(\pi, A, \mu, \sigma^2)$ từ kết quả nhãn cụm.
  3. Chạy thuật toán Baum-Welch (EM) trong 20 vòng lặp:
     * **E-step (`_forward_backward`)**: Tính xác suất Forward $\alpha_t(i)$ và Backward $\beta_t(i)$ hoàn toàn trong không gian Log (`_logsumexp`) để chống tràn số dưới (arithmetic underflow). Tính xác suất hậu nghiệm trạng thái $\gamma_t(i)$ và xác suất chuyển $\xi_t(i, j)$.
     * **M-step (`_m_step`)**: Cập nhật lại tham số $(\pi, A, \mu, \sigma^2)$.
  4. Sắp xếp trạng thái (`_sort_states`): Đảm bảo trạng thái 0 và 1 luôn mang đúng nghĩa vật lý:
     * Đối với mắt (`positive_state="low"`): Sắp xếp sao cho $\mu_0 > \mu_1$ (trạng thái 0 là Mắt Mở - EAR cao, trạng thái 1 là Mắt Nhắm - EAR thấp).
     * Đối với miệng (`positive_state="high"`): Sắp xếp sao cho $\mu_0 < \mu_1$ (trạng thái 0 là Miệng Khép - MAR thấp, trạng thái 1 là Ngáp/Mở - MAR cao).
* **Dự đoán thời gian thực (`predict`)**:
  * Tính tiên nghiệm: $\text{prior} = \text{posterior}_{t-1} \cdot A$.
  * Nhân với mật độ xác suất phát xạ Gauss của giá trị đo hiện tại $P(x_t | S_k)$.
  * Chuẩn hóa Softmax để ra phân phối hậu nghiệm $\text{posterior}_t$. Nhãn trạng thái là $\arg\max(\text{posterior}_t)$.
* **Thích ứng trực tuyến (`update_online`)**:
  * Tích lũy các giá trị quan sát vào hàng đợi đệm kích thước `adapt_interval = 300` khung hình ($\approx 10$ giây ở 30 FPS).
  * Khi đệm đầy, chạy thuật toán EM trên 300 khung hình này và cập nhật tham số theo công thức Exponential Moving Average với tốc độ học $\alpha = 0.01$:
    $$\theta_{new} = (1 - \alpha) \cdot \theta_{old} + \alpha \cdot \theta_{EM}$$
  * Giúp mô hình tự động thích nghi khi tài xế đổi tư thế ngồi, ánh sáng ban ngày chuyển dần sang hoàng hôn, hoặc thay đổi biểu cảm khuôn mặt.

#### B. Lớp `StateMachine` (Máy trạng thái hữu hạn lọc thời gian)
* **Mục đích**: Nhận chuỗi trạng thái nhị phân $(0, 1)$ từ HMM và đo đạc động lực học thời gian:
* **Các ngưỡng thời lượng chuẩn**:
  * `"eye"`: $100 \text{ ms} \le \Delta t \le 2000 \text{ ms}$ (loại bỏ nhiễu $<100$ms, phát hiện nhắm mắt bình thường; nếu $>2000$ms kích hoạt cờ `prolonged` báo ngủ gục).
  * `"mouth"`: $3500 \text{ ms} \le \Delta t \le 7500 \text{ ms}$ (nhận diện cử động ngáp đặc trưng kéo dài từ 3.5s đến 7.5s, bỏ qua nói chuyện hoặc nhai kẹo).
  * `"pitch"`: $800 \text{ ms} \le \Delta t \le 3500 \text{ ms}$ (nhận diện cú gật đầu do ngủ gật, bỏ qua cái gật đầu giao tiếp ngắn $<800$ms).
* **Cơ chế chống mất dấu (Robust Missing Gap Handling)**:
  Nếu khoảng cách thời gian giữa hai khung hình liên tiếp vượt quá `max_gap_sec` ($0.25$s) do tài xế quay hẳn mặt đi hoặc camera bị che, FSM tự động hủy sự kiện dở dang (`self.start = None`) để tránh việc cộng dồn thời gian sai lệch khi khuôn mặt xuất hiện trở lại.
* **Tần suất theo phút (`rate_per_minute`)**:
  Dùng hàng đợi `deque` lưu các mốc thời gian của sự kiện hợp lệ trong 60 giây gần nhất. Tần suất được tính là:
  $$\text{Rate} = \frac{N_{events} \times 60}{\text{WindowSize}}$$

#### C. Lớp `WindowRatio` (Bộ tính tỷ lệ thời gian cửa sổ trượt)
* **Mục đích**: Tính toán tỷ lệ phần trăm thời gian thực mà một trạng thái tích cực diễn ra trong cửa sổ trượt $T_w = 60$ giây.
* **Cách thức hoạt động**:
  * Duy trì danh sách các khoảng thời gian $[t_{start}, t_{end}, \text{active}]$.
  * Khi thời gian trôi đi, tự động loại bỏ các khoảng thời gian nằm ngoài $[t_{now} - T_w, t_{now}]$.
  * Tỷ lệ phần trăm được tính chính xác bằng tích phân thời gian:
    $$\text{Ratio} (\%) = 100 \times \frac{\sum \Delta t_{\text{active}}}{\sum \Delta t_{\text{valid}}}$$
  * Đây là cơ sở toán học để tính **PERCLOS P80**, **POM** và **OverAngle Time**.

#### D. Lớp `AdaptiveHMM_FSM` (Bộ tích hợp thích ứng)
* Tự động điều phối quá trình tự hiệu chuẩn 5 giây đầu (`_initialize`):
  * Lấy phân vị 80% đối với mắt để tìm giá trị mở mắt cơ sở ($\text{EAR}_{normal}$).
  * Thiết lập ngưỡng P80 động:
    $$\text{Threshold}_{P80} = \text{EAR}_{open} - 0.8 \times (\text{EAR}_{open} - \text{EAR}_{closed})$$
  * Tích hợp **chốt an toàn (Safety Guard)**: Nếu $EAR \ge 0.75 \times \text{EAR}_{normal}$, ép trạng thái về 0 (Mắt mở) để loại bỏ hoàn toàn các trường hợp phân loại nhầm khi tài xế đang mở mắt to.

---

### 4.4. `PitchFSM.py` - Mô-đun phân tích động học cúi đầu và phát hiện ngủ gục

#### Mục đích:
File cung cấp giao diện chạy demo độc lập và triển khai thuật toán FSM chuyên sâu cho góc cúi đầu (Pitch). Đây là cơ chế phát hiện trực tiếp hiện tượng **ngủ gật gục đầu (Micro-sleep / Head Nodding)**.

#### Điểm nổi bật trong thuật toán:
1. **So sánh hai phương pháp Head Pose**:
   * Phương pháp 1: `estimate_head_pose_solve_pnp` giải PnP qua OpenCV.
   * Phương pháp 2: `estimate_head_pose_direct_geometry` (Phương pháp hình học trực tiếp).
     Thuật toán hình học trực tiếp sử dụng 5 điểm mốc 3D của MediaPipe (mũi 1, cằm 152, trán 10, mắt trái 263, mắt phải 33):
     $$u_x = \frac{P_{left\_eye} - P_{right\_eye}}{\|P_{left\_eye} - P_{right\_eye}\|}, \quad v_y = \frac{P_{chin} - P_{forehead}}{\|P_{chin} - P_{forehead}\|}$$
     $$u_z = \frac{u_x \times v_y}{\|u_x \times v_y\|}, \quad u_y = u_z \times u_x$$
     Ma trận quay $R = [u_x, u_y, u_z]$. Không cần giải phương trình tối ưu lặp, giúp giảm tối đa tải tính toán CPU/GPU.
2. **Máy trạng thái góc Pitch (Temporal Pitch FSM)**:
   * Ngưỡng góc cúi: `PITCH_DOWN_THRESHOLD = 14.0°`.
   * Nếu thời gian cúi đầu $> 3500$ ms: Phân loại là `CRITICAL: PROLONGED MICRO-SLEEP / DISTRACTION` (Ngủ gục kéo dài nguy hiểm).
   * Nếu thời gian cúi trong khoảng $800 - 3500$ ms: Đánh dấu là một cú gật đầu buồn ngủ hoàn chỉnh (`DROWSY HEAD NOD`), cộng vào bộ đếm tần suất $F_{nod}$.
   * Nếu cúi $< 800$ ms: Đánh dấu là hành động cúi đầu quan sát taplo/đồng hồ xe thông thường (`INTENTIONAL GLANCE`).
3. **Phân cấp cảnh báo nguy cơ**:
   * **CRITICAL (Đỏ)**: Xuất hiện ngủ gục kéo dài $> 3.5$s HOẶC tần suất gật đầu $F_{nod} \ge 3$ lần/phút.
   * **WARNING (Cam)**: Đang có dấu hiệu gật gù HOẶC tần suất $F_{nod} \ge 1$ lần/phút.
   * **NORMAL (Xanh)**: Tư thế lái xe ổn định.

---

### 4.5. `camera_metrics.py` - Bộ điều phối trung tâm (Pipeline Coordinator)

#### Mục đích:
Là file thực thi và tích hợp toàn bộ các mô-đun trên thành một luồng xử lý hoàn chỉnh từ Camera $\to$ Trích xuất đặc trưng $\to$ Phân tích thích ứng $\to$ Xuất dữ liệu $\to$ Hiển thị giao diện HUD.

#### Các thành phần chính:
1. **Lớp `FaceTracker`**:
   Bao bọc mô hình MediaPipe Face Mesh:
   * `max_num_faces = 1` (tập trung duy nhất vào tài xế).
   * `refine_landmarks = True` (bổ sung các điểm mốc chi tiết xung quanh con ngươi và mống mắt).
   * Ngưỡng tin cậy nhận diện và theo dõi: $0.6$.
2. **Lớp `TensorBoardLogger`**:
   Ghi lại toàn bộ chuỗi số liệu thời gian thực (EAR, MAR, Pitch, PERCLOS, Frequencies,...) thành sự kiện TensorBoard để vẽ đồ thị theo dõi và đánh giá mô hình.
3. **Lớp `CameraMetrics`**:
   * **Bộ mốc chỉ số 6 điểm hình học**:
     * Mắt trái: `[362, 385, 387, 263, 373, 380]`
     * Mắt phải: `[33, 160, 158, 133, 153, 144]`
     * Miệng: `[61, 37, 267, 291, 314, 84]`
   * **Hàm tính Tỷ lệ hình học (`aspect_ratio`)**:
     Áp dụng công thức chuẩn Soukupová & Čech:
     $$\text{AR} = \frac{\|P_1 - P_5\| + \|P_2 - P_4\|}{2 \cdot \|P_0 - P_3\|}$$
   * **Hiệu chuẩn tư thế trung hòa tự động (Neutral Pose Calibration)**:
     * Thu thập 30 khung hình đầu tiên có khuôn mặt hợp lệ.
     * Tính trung vị (median) của 3 góc (Pitch, Yaw, Roll) làm mốc chuẩn $\theta_{neutral}$.
     * Chuẩn hóa góc quay bù góc:
       $$\Delta \theta = (\theta_{raw} - \theta_{neutral} + 180^\circ) \pmod{360^\circ} - 180^\circ$$
   * **Cơ chế trễ phát hiện gật đầu (Hysteresis Nod Detection)**:
     Để tránh hiện tượng cờ trạng thái bị bật/tắt liên tục (chattering) khi góc cúi dao động quanh ngưỡng, hệ thống dùng hai ngưỡng khác nhau:
     * Kích hoạt cúi đầu khi: $\text{pitch} > 14^\circ$.
     * Giải phóng trạng thái cúi đầu khi: $\text{pitch} < 8^\circ$.
   * **Kiểm tra góc lệch giới hạn (Over-Angle)**:
     Ngưỡng an toàn cho phép: $|\text{Pitch}| \le 20^\circ$, $|\text{Yaw}| \le 25^\circ$, $|\text{Roll}| \le 20^\circ$. Vượt quá các góc này tức là tài xế đang quay mặt đi nơi khác, được đưa vào `WindowRatio` để tính % thời gian lơ là.
   * **Vẽ giao diện bảng thông số HUD (`draw`)**:
     Tạo bảng hiển thị bên cạnh khung hình camera với đầy đủ trạng thái phân màu:
     * Khu vực EYES: EAR, Eye state (OPEN/CLOSED), PERCLOS P80 %, Blink rate, Thời gian nhắm.
     * Khu vực MOUTH: MAR, Mouth state (RESTING/OPEN), POM %, Yawn frequency, Thời gian mở miệng.
     * Khu vực HEAD: Pitch, Yaw, Roll, Over-angle %, Nodding status, Nod frequency, Nod duration.
     * Thông số hệ thống: FPS tức thời và độ trễ xử lý mỗi khung hình (Processing latency tính bằng ms).
   * **Vòng lặp chạy (`run`)**:
     Quản lý tài nguyên camera với `contextlib.ExitStack`, hỗ trợ xử lý video offline theo đúng mốc thời gian file (`use_video_time`), định kỳ xuất JSON theo chu kỳ `output_interval_sec` (mặc định 1 giây/lần), và cung cấp hook `on_metrics` để truyền dữ liệu trực tiếp tới các luồng khác.

---

## 5. BẢNG TỔNG HỢP CÁC CHỈ SỐ VÀ NGƯỠNG PHÂN LOẠI (METRICS SPECIFICATION)

| Chỉ số (Metric) | Tên trường (Field) | Phạm vi / Đơn vị | Phương pháp tính toán | Ý nghĩa an toàn giao thông |
| :--- | :--- | :--- | :--- | :--- |
| **EAR** | `ear` | $0.15 - 0.40$ (vô thứ nguyên) | Tỷ lệ hình học 6 điểm mắt | Đo độ mở của mắt tức thời |
| **Trạng thái mắt** | `eye_state` | `0` (Mở), `1` (Đóng) | AdaptiveHMM (học trực tuyến) | Phân loại trạng thái mắt thích ứng từng người |
| **Tần suất chớp mắt** | `blink_rate_per_min` | $10 - 40$ lần/phút | StateMachine (cửa sổ 60s) | Tần suất quá thấp hoặc quá cao đều cảnh báo mệt mỏi |
| **Thời gian nhắm mắt** | `eye_closure_duration_ms` | mili-giây (ms) | FSM tích lũy thời gian | $> 2000$ ms: Ngủ gật nguy cấp (Prolonged closure) |
| **PERCLOS (P80)** | `perclos_pct` | $0.0 - 100.0$ (%) | Tích phân thời gian WindowRatio | **Tiêu chuẩn vàng**: $\ge 8 - 12\%$ cảnh báo buồn ngủ |
| **MAR** | `mar` | $0.10 - 0.90$ (vô thứ nguyên) | Tỷ lệ hình học 6 điểm miệng | Đo độ mở của miệng tức thời |
| **Trạng thái miệng** | `mouth_state` | `0` (Khép), `1` (Mở) | AdaptiveHMM (học trực tuyến) | Nhận diện cử động há miệng |
| **Tần suất ngáp** | `yawning_frequency_per_min`| lần/phút | StateMachine ($3.5s \le \Delta t \le 7.5s$) | Dấu hiệu sinh lý của sự thiếu ngủ/uể oải |
| **Thời gian mở miệng** | `mouth_open_duration_ms` | mili-giây (ms) | FSM tích lũy thời gian | Thời gian ngáp thực tế |
| **POM** | `pom_pct` | $0.0 - 100.0$ (%) | Tích phân thời gian WindowRatio | Tỷ lệ thời gian miệng mở trong 1 phút |
| **Góc Pitch** | `pitch_deg` | độ ($^\circ$) | SolvePnP / RQDecomp - bù trung hòa | Cúi đầu ($+$) hoặc ngửa đầu ($-$) so với hướng nhìn |
| **Góc Yaw** | `yaw_deg` | độ ($^\circ$) | SolvePnP / RQDecomp - bù trung hòa | Quay sang trái ($+$) hoặc sang phải ($-$) |
| **Góc Roll** | `roll_deg` | độ ($^\circ$) | SolvePnP / RQDecomp - bù trung hòa | Nghiêng đầu sang trái ($+$) hoặc phải ($-$) |
| **Tần suất gật gù** | `nodding_frequency_per_min`| lần/phút | StateMachine ($800ms \le \Delta t \le 3500ms$)| Phát hiện hiện tượng gật gù do buồn ngủ ($F_{nod}$) |
| **Thời gian cúi đầu** | `nod_duration_ms` | mili-giây (ms) | FSM tích lũy thời gian | $> 3500$ ms: Ngủ gục sâu hoặc mất tập trung nguy hiểm |
| **Tỷ lệ lệch hướng** | `over_angle_pct` | $0.0 - 100.0$ (%) | WindowRatio ($>$ limits) | Tỷ lệ thời gian không nhìn đường phía trước |

---

## 6. HƯỚNG DẪN SỬ DỤNG VÀ CHẠY THỬ NGHIỆM (USAGE GUIDE)

### 6.1. Chạy giám sát Camera thời gian thực
Khởi chạy bộ điều phối trung tâm với webcam mặc định, hiển thị HUD và ghi dữ liệu:
```bash
python -m ai.PhysicalBranch.camera_metrics
```
* Phím `r`: Đặt lại quá trình hiệu chuẩn tư thế trung hòa và lịch sử cửa sổ trượt.
* Phím `q`: Thoát chương trình.

### 6.2. Chạy thử nghiệm phân tích góc Pitch và gật đầu FSM
```bash
python -m ai.PhysicalBranch.PitchFSM
```
* Phím `m`: Chuyển đổi giữa 2 phương pháp ước lượng (`OpenCV SolvePnP` $\leftrightarrow$ `Direct 3D Geometry`).
* Phím `r`: Hiệu chuẩn góc nhìn thẳng hiện tại.
* Phím `q`: Thoát.

### 6.3. Tích hợp theo dạng thư viện Python (Programmatic Usage)
```python
import cv2
from ai.PhysicalBranch.camera_metrics import CameraMetrics
from ai.PhysicalBranch.interface import AIResult

# 1. Khởi tạo bộ giám sát
monitor = CameraMetrics(fps=30, window_sec=60)

# 2. Xử lý từng khung hình
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    metrics = monitor.process(frame)
    
    # 3. Đóng gói sang đối tượng AIResult chuẩn hóa
    result = AIResult(
        timestamp_sec=metrics["timestamp_sec"],
        window_sec=60.0,
        ear=metrics["ear"],
        blink_rate_per_min=metrics["blink_rate_per_min"],
        eye_closure_duration_ms=metrics["eye_closure_duration_ms"],
        perclos_pct=metrics["perclos_pct"],
        mar=metrics["mar"],
        mouth_open_duration_ms=metrics["mouth_open_duration_ms"],
        pitch_deg=metrics["pitch_deg"],
        pitch_deviation_duration_ms=metrics["nod_duration_ms"],
        pitch_deviation_rate_per_min=metrics["nodding_frequency_per_min"]
    )
    
    # 4. Xuất chuỗi JSON gửi cho backend hoặc MQTT
    json_data = result.to_json()
    # print(json_data)

cap.release()
monitor.close()
```

---

## 7. MỐI QUAN HỆ VỚI BẢN KOTLIN (`PhysicalBranch_kotlin`)

Toàn bộ thuật toán toán học trong thư mục này đã được đối chiếu và chuyển đổi song song sang ngôn ngữ **Kotlin** thuần trong thư mục `ai/PhysicalBranch_kotlin`:
* `AdaptiveHMM` $\to$ `AdaptiveHMM.kt` (Triển khai toán học thuần không cần phụ thuộc thư viện native).
* `StateMachine` $\to$ `StateMachine.kt`.
* `WindowRatio` $\to$ `WindowRatio.kt`.
* `HeadPoseEstimator` $\to$ `HeadPoseEstimator.kt` (Hỗ trợ giải thuật Direct 3D Geometry).
* `PitchFSM.py` $\to$ `PitchFsm.kt`.
* `camera_metrics.py` $\to$ `CameraMetrics.kt`.
* `interface.py` $\to$ `AIResult.kt`.

Sự đồng bộ này cho phép các nghiên cứu, kiểm chứng thuật toán và huấn luyện được thực hiện nhanh chóng trên Python, sau đó đưa thẳng mã nguồn Kotlin lên hệ điều hành trên xe hơi (**Android Automotive OS**) mà không cần thay đổi logic thuật toán.
