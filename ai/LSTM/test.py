from src.models import ConvGRUClassifier
from configs.config import TrainConfig, load_config

cfg = load_config(config_path=r"D:\Project\DATN\driver-guardian\ai\LSTM\configs\config.yaml")

model = ConvGRUClassifier(
    input_dim=64,
    hidden_dim=128
)
total_params = sum(p.numel() for p in model.parameters())

