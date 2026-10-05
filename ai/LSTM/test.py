from src.models1 import ConvGRUClassifier

model = ConvGRUClassifier()
total_params = sum(p.numel() for p in model.parameters())

# Tổng số tham số cần train (requires_grad = True)
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

# Tổng số tham số bị đóng băng (non-trainable)
non_trainable_params = total_params - trainable_params

print(f"Tổng số tham số:        {total_params:,}")
print(f"Tham số cần train:      {trainable_params:,}")
print(f"Tham số không train:    {non_trainable_params:,}")
