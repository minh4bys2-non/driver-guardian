from src.models import ConvGRUClassifier
from configs.config import TrainConfig, load_config

cfg = load_config(config_path=r"D:\Project\DATN\driver-guardian\ai\LSTM\configs\config.yaml")

model = ConvGRUClassifier.from_config(r"D:\Project\DATN\driver-guardian\ai\checkpoints\model_convgru\best.pt")

print(model)
