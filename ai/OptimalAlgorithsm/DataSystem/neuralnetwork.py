import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # x sẽ có dạng mảng numpy [num_frame, 3, H, W]
        # Tổng số lượng frame: 30fps * từ 10 - 15 giây
        # Cụ thể: [Frame: RGB, shape: [3, H, W], dtype: unit8, range: [0, 255], timeline = 30FPS]

        # Yêu cầu đầu ra: 1 số duy nhất thể hiện mức độ buồn ngủ.
        return 0.5