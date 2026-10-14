import math
import sys
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvGRUCell(nn.Module):
    """
    Tế bào Convolutional GRU đơn tầng (Single-layer ConvGRU Cell).

    Tính toán trạng thái ẩn tiếp theo H_t từ đặc trưng đầu vào X_t và trạng thái ẩn trước H_{t-1}
    thông qua các phép tích chập 2 chiều bảo toàn kích thước không gian.

    Hệ phương trình:
        R_t = sigmoid(W_r * [X_t, H_{t-1}] + b_r)                  # Cổng Reset
        Z_t = sigmoid(W_z * [X_t, H_{t-1}] + b_z)                  # Cổng Update
        H~_t = tanh(W_h * [X_t, R_t * H_{t-1}] + b_h)              # Trạng thái ứng viên
        H_t = (1 - Z_t) * H_{t-1} + Z_t * H~_t                     # Cập nhật trạng thái
    """

    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            kernel_size: Union[int, Tuple[int, int]] = 3,
            bias: bool = True,
            padding_mode: str = "zeros"
    ):
        """
        Khởi tạo ConvGRUCell.

        Args:
            input_dim (int): Số kênh (channels) của tensor đầu vào X_t.
            hidden_dim (int): Số kênh của trạng thái ẩn H_t.
            kernel_size (int | Tuple[int, int]): Kích thước kernel tích chập (mặc định 3).
            bias (bool): Có sử dụng vector độ lệch bias hay không (mặc định True).
            padding_mode (str): Chế độ padding ('zeros', 'reflect', 'replicate').
        """
        super().__init__()

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.bias = bias

        # Chuẩn hóa kernel_size và padding
        if isinstance(kernel_size, int):
            self.kernel_size: Tuple[int, int] = (kernel_size, kernel_size)
        else:
            self.kernel_size = tuple(kernel_size)

        self.padding: Tuple[int, int] = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)

        # 1. Kernel Fusion: Gộp 2 cổng Reset và Update vào một tầng Conv2D duy nhất
        # Đầu vào: [X_t, H_{t-1}] có số kênh = input_dim + hidden_dim
        # Đầu ra: [R_t, Z_t] có số kênh = 2 * hidden_dim
        self.conv_gates = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=2 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            padding_mode=padding_mode,
            bias=self.bias
        )

        # 2. Tầng tích chập tính toán Trạng thái ứng viên (Candidate state H~_t)
        # Đầu vào: [X_t, R_t * H_{t-1}] có số kênh = input_dim + hidden_dim
        # Đầu ra: H~_t có số kênh = hidden_dim
        self.conv_candidate = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            padding_mode=padding_mode,
            bias=self.bias
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Khởi tạo trọng số chuẩn cho mô hình hồi quy tích chập:
          - Khởi tạo trực giao (Orthogonal) cho phần trọng số phản hồi (recurrent weights).
          - Khởi tạo Kaiming Normal cho phần trọng số đầu vào (input weights).
          - Khởi tạo bias: zeros cho reset/candidate, và khởi tạo nhẹ dương cho update gate
            để hỗ trợ ghi nhớ trạng thái dài hạn ở giai đoạn đầu huấn luyện.
        """
        # --- Khởi tạo conv_gates ---
        with torch.no_grad():
            w_gates = self.conv_gates.weight
            # Tách phần input và recurrent theo chiều input channels
            w_gates_x = w_gates[:, :self.input_dim, :, :]
            w_gates_h = w_gates[:, self.input_dim:, :, :]

            nn.init.kaiming_normal_(w_gates_x, mode="fan_out", nonlinearity="relu")
            # Khởi tạo trực giao độc lập cho từng cổng Reset và Update để phá vỡ hoàn toàn tính liên đới đối xứng
            nn.init.orthogonal_(w_gates_h[:self.hidden_dim])
            nn.init.orthogonal_(w_gates_h[self.hidden_dim:])

            if self.conv_gates.bias is not None:
                nn.init.zeros_(self.conv_gates.bias)
                # Đặt bias của update gate (nửa sau) thành 1.0 để giữ nguyên thông tin ban đầu
                self.conv_gates.bias[self.hidden_dim:].fill_(1.0)

            # --- Khởi tạo conv_candidate ---
            w_can = self.conv_candidate.weight
            w_can_x = w_can[:, :self.input_dim, :, :]
            w_can_h = w_can[:, self.input_dim:, :, :]

            nn.init.kaiming_normal_(w_can_x, mode="fan_out", nonlinearity="tanh")
            nn.init.orthogonal_(w_can_h)

            if self.conv_candidate.bias is not None:
                nn.init.zeros_(self.conv_candidate.bias)

    def init_hidden(
            self,
            batch_size: int,
            spatial_size: Tuple[int, int],
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None
    ) -> torch.Tensor:
        """
        Khởi tạo trạng thái ẩn toàn số 0 động theo batch_size, kích thước không gian và thiết bị.

        Args:
            batch_size (int): Kích thước batch.
            spatial_size (Tuple[int, int]): (Chiều cao H, Chiều rộng W).
            device (torch.device, optional): Thiết bị cấp phát tensor (mặc định lấy từ tham số mô hình).
            dtype (torch.dtype, optional): Kiểu dữ liệu tensor (mặc định lấy từ tham số mô hình).

        Returns:
            torch.Tensor: Tensor trạng thái ẩn [Batch, hidden_dim, H, W].
        """
        target_device = device if device is not None else self.conv_gates.weight.device
        target_dtype = dtype if dtype is not None else self.conv_gates.weight.dtype

        return torch.zeros(
            batch_size,
            self.hidden_dim,
            spatial_size[0],
            spatial_size[1],
            device=target_device,
            dtype=target_dtype
        )

    def forward(
            self,
            input_tensor: torch.Tensor,
            h_cur: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Lan truyền xuôi một bước thời gian đơn lẻ.

        Args:
            input_tensor (torch.Tensor): Đặc trưng đầu vào [Batch, input_dim, H, W].
            h_cur (torch.Tensor, optional): Trạng thái ẩn trước đó [Batch, hidden_dim, H, W].
                Nếu None, tự động khởi tạo zeros theo kích thước của input_tensor.

        Returns:
            torch.Tensor: Trạng thái ẩn kế tiếp H_next [Batch, hidden_dim, H, W].
        """
        if input_tensor.dim() != 4:
            raise ValueError(
                f"[ConvGRUCell] Yêu cầu input_tensor 4D [Batch, Channels, H, W], "
                f"nhưng nhận được tensor {input_tensor.dim()}D với kích thước {list(input_tensor.shape)}."
            )

        batch_size, _, height, width = input_tensor.shape

        if h_cur is None:
            h_cur = self.init_hidden(
                batch_size=batch_size,
                spatial_size=(height, width),
                device=input_tensor.device,
                dtype=input_tensor.dtype
            )
        else:
            if h_cur.shape != (batch_size, self.hidden_dim, height, width):
                raise ValueError(
                    f"[ConvGRUCell] Kích thước h_cur {list(h_cur.shape)} không khớp với "
                    f"kỳ vọng {(batch_size, self.hidden_dim, height, width)}."
                )

        # 1. Tính toán song song cổng Reset và cổng Update qua Kernel Fusion
        combined = torch.cat([input_tensor, h_cur], dim=1)
        gates = self.conv_gates(combined)
        reset_raw, update_raw = torch.split(gates, self.hidden_dim, dim=1)

        reset_gate = torch.sigmoid(reset_raw)
        update_gate = torch.sigmoid(update_raw)

        # Đảm bảo kiểu dữ liệu tương thích chuẩn xác kể cả khi chạy Mixed Precision (AMP Autocast)
        h_cur_casted = h_cur.to(dtype=reset_gate.dtype)
        in_casted = input_tensor.to(dtype=reset_gate.dtype)

        # 2. Tính toán trạng thái ứng viên (Candidate Hidden State)
        combined_candidate = torch.cat([in_casted, reset_gate * h_cur_casted], dim=1)
        candidate_raw = self.conv_candidate(combined_candidate)
        candidate = torch.tanh(candidate_raw)

        # 3. Cập nhật trạng thái ẩn theo pha trộn lồi bảo toàn không gian
        h_next = (1.0 - update_gate) * h_cur_casted + update_gate * candidate
        return h_next


class ConvGRU(nn.Module):
    """
    Module Convolutional GRU Đa Tầng (Multi-layer Spatio-Temporal ConvGRU).

    Hỗ trợ xử lý toàn diện chuỗi thời gian video không gian 5D:
      - Nhận đầu vào 5D dạng `[B, T, C, H, W]` (hoặc `[T, B, C, H, W]`).
      - Hỗ trợ Dynamic Spatial Size: Tự thích ứng với mọi độ phân giải đầu vào.
      - Hỗ trợ Stateful Streaming Inference: Cho phép truyền và nhận `hidden_state` ngoại vi.
      - Hỗ trợ Dynamic Sequence Masking (`seq_lens`): Đóng băng trạng thái ẩn tại các frame padding.
      - Hỗ trợ Spatial Dropout giữa các tầng và Residual Connection tùy chọn.
    """

    def __init__(
            self,
            input_dim: int,
            hidden_dim: Union[int, List[int], Tuple[int, ...]],
            kernel_size: Union[int, Tuple[int, int], List[Union[int, Tuple[int, int]]]] = 3,
            num_layers: int = 1,
            batch_first: bool = True,
            bias: bool = True,
            return_all_layers: bool = False,
            dropout: float = 0.0,
            residual: bool = False
    ):
        """
        Khởi tạo ConvGRU đa tầng.

        Args:
            input_dim (int): Số kênh đặc trưng của tensor đầu vào.
            hidden_dim (int | List[int]): Số kênh ẩn cho từng tầng (nếu truyền int, áp dụng cho mọi tầng).
            kernel_size (int | Tuple | List): Kích thước kernel cho từng tầng.
            num_layers (int): Số lượng tầng ConvGRU xếp chồng.
            batch_first (bool): Nếu True, tensor đầu vào có dạng [B, T, C, H, W]; nếu False là [T, B, C, H, W].
            bias (bool): Có thêm bias hay không.
            return_all_layers (bool): Nếu True, trả về output của tất cả các tầng; nếu False, chỉ trả về tầng cuối.
            dropout (float): Tỷ lệ Spatial Dropout2d giữa các tầng (chỉ áp dụng khi num_layers > 1).
            residual (bool): Bật kết nối tắt (Residual/Skip Connection) giữa các tầng có cùng số kênh.
        """
        super().__init__()

        self.input_dim = int(input_dim)
        self.num_layers = int(num_layers)
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers
        self.residual = residual

        # 1. Chuẩn hóa hidden_dim cho từng tầng
        if isinstance(hidden_dim, (list, tuple)):
            if len(hidden_dim) != self.num_layers:
                raise ValueError(
                    f"[ConvGRU] Chiều dài danh sách hidden_dim ({len(hidden_dim)}) "
                    f"phải khớp với num_layers ({self.num_layers})."
                )
            self.hidden_dims = [int(h) for h in hidden_dim]
        else:
            self.hidden_dims = [int(hidden_dim)] * self.num_layers

        # 2. Chuẩn hóa kernel_size cho từng tầng
        if isinstance(kernel_size, list):
            if len(kernel_size) != self.num_layers:
                raise ValueError(
                    f"[ConvGRU] Chiều dài danh sách kernel_size ({len(kernel_size)}) "
                    f"phải khớp với num_layers ({self.num_layers})."
                )
            self.kernel_sizes = kernel_size
        else:
            self.kernel_sizes = [kernel_size] * self.num_layers

        # 3. Khởi tạo danh sách các cell bằng nn.ModuleList chuẩn mực
        cell_list: List[nn.Module] = []
        dropout_list: List[nn.Module] = []

        for layer_idx in range(self.num_layers):
            cur_in_dim = self.input_dim if layer_idx == 0 else self.hidden_dims[layer_idx - 1]
            cur_hid_dim = self.hidden_dims[layer_idx]
            cur_k_size = self.kernel_sizes[layer_idx]

            cell = ConvGRUCell(
                input_dim=cur_in_dim,
                hidden_dim=cur_hid_dim,
                kernel_size=cur_k_size,
                bias=self.bias
            )
            cell_list.append(cell)

            # Spatial Dropout áp dụng giữa các tầng trung gian
            if dropout > 0.0 and layer_idx < self.num_layers - 1:
                dropout_list.append(nn.Dropout2d(p=dropout))
            else:
                dropout_list.append(nn.Identity())

        self.cells = nn.ModuleList(cell_list)
        self.dropouts = nn.ModuleList(dropout_list)

    def init_hidden(
            self,
            batch_size: int,
            spatial_size: Tuple[int, int],
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None
    ) -> List[torch.Tensor]:
        """
        Khởi tạo trạng thái ẩn toàn số 0 cho tất cả các tầng.

        Args:
            batch_size (int): Kích thước batch.
            spatial_size (Tuple[int, int]): (H, W).
            device (torch.device, optional): Thiết bị cấp phát.
            dtype (torch.dtype, optional): Kiểu dữ liệu.

        Returns:
            List[torch.Tensor]: Danh sách hidden state ban đầu cho num_layers tầng.
        """
        return [
            cell.init_hidden(batch_size, spatial_size, device=device, dtype=dtype)
            for cell in self.cells
        ]

    def forward(
            self,
            input_tensor: torch.Tensor,
            hidden_state: Optional[List[torch.Tensor]] = None,
            seq_lens: Optional[torch.Tensor] = None
    ) -> Tuple[Union[torch.Tensor, List[torch.Tensor]], List[torch.Tensor]]:
        """
        Lan truyền xuôi chuỗi thời gian không gian.

        Args:
            input_tensor (torch.Tensor): Tensor chuỗi 5D:
                - Nếu batch_first=True: [Batch, Time, Channels, Height, Width]
                - Nếu batch_first=False: [Time, Batch, Channels, Height, Width]
            hidden_state (List[torch.Tensor], optional): Trạng thái ẩn ban đầu từ bên ngoài
                (dùng cho streaming inference liên tục giữa các chunk). Nếu None, tự khởi tạo zeros.
            seq_lens (torch.Tensor, optional): Tensor 1D chứa độ dài thực tế của từng clip [Batch].
                Các khung hình tại t >= seq_lens sẽ được đóng băng trạng thái ẩn (không cập nhật rác).

        Returns:
            Tuple[output, last_states]:
              - output:
                  + Nếu return_all_layers=False: Tensor 5D của tầng cuối [Batch, Time, H_last, H, W].
                  + Nếu return_all_layers=True: Danh sách các Tensor 5D của tất cả các tầng.
              - last_states: Danh sách các Tensor 4D [Batch, H_dim, H, W] đại diện cho trạng thái ẩn
                cuối cùng của từng tầng (dùng để truyền tiếp cho chunk tiếp theo trong streaming).
        """
        if input_tensor.dim() != 5:
            raise ValueError(
                f"[ConvGRU] Yêu cầu input_tensor 5D [Batch, Time, C, H, W] hoặc [Time, Batch, C, H, W], "
                f"nhưng nhận được {input_tensor.dim()}D với kích thước {list(input_tensor.shape)}."
            )

        # Chuyển về định dạng batch_first [B, T, C, H, W] để xử lý đồng nhất
        if not self.batch_first:
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        batch_size, seq_len, in_channels, height, width = input_tensor.shape

        if in_channels != self.input_dim:
            raise ValueError(
                f"[ConvGRU] Số kênh đầu vào ({in_channels}) không khớp với cấu hình input_dim ({self.input_dim})."
            )

        # Khởi tạo hoặc kiểm tra tính hợp lệ của hidden_state truyền vào
        if hidden_state is None:
            cur_states = self.init_hidden(
                batch_size=batch_size,
                spatial_size=(height, width),
                device=input_tensor.device,
                dtype=input_tensor.dtype
            )
        else:
            if len(hidden_state) != self.num_layers:
                raise ValueError(
                    f"[ConvGRU] Số lượng hidden_state ({len(hidden_state)}) "
                    f"không khớp với num_layers ({self.num_layers})."
                )
            cur_states = list(hidden_state)

        # Chuẩn bị mặt nạ masking nếu có seq_lens
        masks: Optional[torch.Tensor] = None
        if seq_lens is not None:
            # Tạo boolean mask dạng [Batch, Time]
            time_indices = torch.arange(seq_len, device=input_tensor.device).unsqueeze(0)  # [1, T]
            seq_lens_tensor = seq_lens.to(input_tensor.device).unsqueeze(1)  # [B, 1]
            masks = (time_indices < seq_lens_tensor)  # [B, T]

        layer_outputs: List[torch.Tensor] = []
        last_states: List[torch.Tensor] = []
        cur_layer_input = input_tensor

        # Lặp qua từng tầng kiến trúc
        for layer_idx in range(self.num_layers):
            cell = self.cells[layer_idx]
            dropout_layer = self.dropouts[layer_idx]
            h_t = cur_states[layer_idx]

            time_outputs: List[torch.Tensor] = []

            # Lặp qua từng bước thời gian t trong chuỗi
            for t in range(seq_len):
                x_t = cur_layer_input[:, t, :, :, :]  # [B, C, H, W]
                h_next = cell(x_t, h_t)  # [B, H_dim, H, W]

                # Nếu có sequence masking, đóng băng trạng thái khi vượt quá seq_len
                if masks is not None:
                    mask_t = masks[:, t].view(batch_size, 1, 1, 1).to(dtype=h_next.dtype)
                    h_t = mask_t * h_next + (1.0 - mask_t) * h_t.to(dtype=h_next.dtype)
                else:
                    h_t = h_next

                time_outputs.append(h_t)

            # Gom tụ đầu ra thời gian: [B, T, H_dim, H, W]
            layer_output = torch.stack(time_outputs, dim=1)

            # Áp dụng Residual Connection nếu được yêu cầu và số kênh khớp nhau
            if self.residual and cur_layer_input.shape[2] == layer_output.shape[2]:
                layer_output = layer_output + cur_layer_input.to(dtype=layer_output.dtype)

            # Áp dụng Spatial Dropout giữa các tầng (nếu có)
            if not isinstance(dropout_layer, nn.Identity):
                # Reshape tạm để đưa qua Dropout2d: [B*T, C, H, W]
                b_sz, t_sz, c_sz, h_sz, w_sz = layer_output.shape
                layer_output_reshaped = layer_output.view(b_sz * t_sz, c_sz, h_sz, w_sz)
                layer_output = dropout_layer(layer_output_reshaped).view(b_sz, t_sz, c_sz, h_sz, w_sz)

            cur_layer_input = layer_output
            layer_outputs.append(layer_output)
            last_states.append(h_t)

        # Chuyển đổi lại shape nếu người dùng cấu hình batch_first=False
        if not self.batch_first:
            layer_outputs = [out.permute(1, 0, 2, 3, 4) for out in layer_outputs]

        final_output = layer_outputs if self.return_all_layers else layer_outputs[-1]
        return final_output, last_states


if __name__ == "__main__":
    pass
