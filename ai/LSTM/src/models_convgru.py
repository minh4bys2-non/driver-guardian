import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvGRUCell(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=64, kernel_size=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        channels = input_dim + hidden_dim
        self.gates = nn.Conv2d(channels, 2 * hidden_dim, kernel_size, padding=kernel_size // 2)
        self.candidate = nn.Conv2d(channels, hidden_dim, kernel_size, padding=kernel_size // 2)

    def forward(self, x, h):
        reset, update = self.gates(torch.cat([x, h], dim=1)).sigmoid().chunk(2, dim=1)
        candidate = self.candidate(torch.cat([x, reset * h], dim=1)).tanh()
        return (1 - update) * h + update * candidate

class ConvGRUClassifier(nn.Module):
    def __init__(
        self,
        input_dim=96,
        hidden_dim=96,
        num_classes=2,
        spatial_in_channels=(64, 128, 256),
        kernel_size=3,
        dropout=0.30,
        supervision_mode="attention_pooling",
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_classes = int(num_classes)
        self.supervision_mode = supervision_mode.lower()
        self.p4_channels = int(spatial_in_channels[1])

        self.adapter = nn.Sequential(
            nn.Conv2d(self.p4_channels, self.input_dim, 1, bias=False),
            nn.GroupNorm(1, self.input_dim),
            nn.SiLU(),
        )
        self.convgru = ConvGRUCell(self.input_dim, self.hidden_dim, kernel_size)
        self.spatial_attention = nn.Conv2d(self.hidden_dim, 1, 1)
        self.fc_out = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.hidden_dim, self.num_classes))

        self.init_weight()
    def init_weight(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                if module is self.adapter[0]:
                    nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
                else:
                    nn.init.xavier_uniform_(module.weight)

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        nn.init.constant_(self.convgru.gates.bias[self.hidden_dim:], -1.0)
    @classmethod
    def from_checkpoint(cls, checkpoint_path, map_location="cpu"):
        from .checkpoint import load_model
        model, _ = load_model(checkpoint_path, map_location)
        return model.conv_gru

    def forward(
        self,
        features,
        seq_lens=None,
        hidden_state=None,
        return_sequence=False,
        return_weights=False,
        return_state=False,
    ):
        if isinstance(features, (tuple, list)):
            if len(features) != 3:
                raise ValueError("Đầu vào multi-scale phải gồm (P3, P4, P5).")
            features = features[1]

        single_frame = features.ndim == 4
        if single_frame:
            features = features.unsqueeze(1)
        if features.ndim != 5 or any(size == 0 for size in features.shape):
            raise ValueError("Expected nonempty [B,T,C,H,W] or [B,C,H,W].")

        b, t, c, height, width = features.shape
        if c == self.p4_channels:
            features = self.adapter(features.reshape(b * t, c, height, width))
            features = features.reshape(b, t, self.input_dim, height, width)
        elif c != self.input_dim:
            raise ValueError(f"Expected {self.p4_channels} P4 channels or {self.input_dim} adapted channels; got {c}.")

        if seq_lens is None:
            valid = torch.ones(b, t, dtype=torch.bool, device=features.device)
        else:
            seq_lens = torch.as_tensor(seq_lens, device=features.device)
            if seq_lens.shape != (b,) or not torch.all((seq_lens >= 1) & (seq_lens <= t) & (seq_lens == seq_lens.long())):
                raise ValueError("seq_lens must have shape [B] and integer values in [1,T].")
            valid = torch.arange(t, device=features.device)[None] < seq_lens[:, None]

        if hidden_state is None:
            h = features.new_zeros((b, self.hidden_dim, height, width))
        else:
            h = hidden_state[0] if isinstance(hidden_state, (tuple, list)) else hidden_state
            if h.shape != (b, self.hidden_dim, height, width):
                raise ValueError("Invalid hidden_state shape.")

        vectors, spatial_weights = [], []
        for i in range(t):
            new_h = self.convgru(features[:, i], h)
            h = torch.where(valid[:, i, None, None, None], new_h, h)
            attention = F.softmax(self.spatial_attention(h).flatten(2), dim=-1)
            attention = attention.reshape(b, 1, height, width)
            vectors.append((h * attention).sum(dim=(-2, -1)))
            if return_weights:
                spatial_weights.append(attention * valid[:, i, None, None, None])

        vectors = torch.stack(vectors, dim=1)
        sequence_mode = return_sequence or self.supervision_mode in ("sequence", "frame")
        temporal_weights = None
        if sequence_mode:
            logits = self.fc_out(vectors)
            logits = logits.masked_fill(~valid.unsqueeze(-1), 0)
            if single_frame:
                logits = logits.squeeze(1)
        else:
            temporal_weights = valid.to(vectors.dtype)
            temporal_weights = temporal_weights / temporal_weights.sum(dim=1, keepdim=True)
            logits = self.fc_out((vectors * temporal_weights.unsqueeze(-1)).sum(dim=1))

        output = [logits]
        if return_weights:
            output.append({"spatial": torch.stack(spatial_weights, dim=1), "temporal": temporal_weights})
        if return_state:
            output.append([h])
        return tuple(output) if len(output) > 1 else logits

if __name__ == "__main__":
    torch.manual_seed(42)
    model = ConvGRUClassifier().eval()
    print("P4 -> adapter -> ConvGRU -> spatial attention -> temporal mean -> classifier")
    for name, module in model.named_children():
        print(f"{name:20s}: {sum(p.numel() for p in module.parameters()):,} parameters")
    print(f"{'Total':20s}: {sum(p.numel() for p in model.parameters()):,} parameters")

    p4 = torch.randn(2, 4, model.p4_channels, 14, 14)
    seq_lens = torch.tensor([4, 2])
    with torch.inference_mode():
        logits, weights, states = model(p4, seq_lens=seq_lens, return_weights=True, return_state=True)
    for name, tensor in (
        ("P4", p4), ("Logits", logits), ("Spatial attention", weights["spatial"]),
        ("Temporal weights", weights["temporal"]), ("Hidden state", states[0]),
    ):
        print(f"{name:20s}: {tuple(tensor.shape)}")
