import math

import torch
import torch.nn as nn
import torch.nn.functional as F

def autopad(k, p=None, d=1):
    if p is not None:
        return p
    return (d * (k - 1) + 1) // 2

class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(
            c1, c2, k, s, autopad(k, p, d),
            groups=g, dilation=d, bias=False,
        )
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class DWConv(Conv):
    def __init__(self, c1, c2, k=1, s=1, act=True):
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), act=act)

class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        if c_ < 1:
            raise ValueError("int(c2 * e) must be positive")
        self.cv1 = Conv(c1, c_, 3, 1)
        self.cv2 = Conv(c_, c2, 3, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y

class C2f(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        if self.c < 1 or n < 0:
            raise ValueError("int(c2 * e) must be positive and n nonnegative")
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1, 1)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, e=1.0) for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        for m in self.m:
            y.append(m(y[-1]))
        return self.cv2(torch.cat(y, 1))

class CIB(nn.Module):
    def __init__(self, c1, c2, shortcut=True, e=0.5, large_kernel=False):
        super().__init__()
        c_ = int(c2 * e)
        if c_ < 1:
            raise ValueError("int(c2 * e) must be positive")
        k = 7 if large_kernel else 3
        self.block = nn.Sequential(
            Conv(c1, c1, k, 1, g=c1),
            Conv(c1, 2 * c_, 1, 1),
            Conv(2 * c_, 2 * c_, k, 1, g=2 * c_),
            Conv(2 * c_, c2, 1, 1),
            Conv(c2, c2, k, 1, g=c2),
        )
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.block(x)
        return x + y if self.add else y

class C2fCIB(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, e=0.5, large_kernel=False):
        super().__init__(c1, c2, n, shortcut, e)
        self.m = nn.ModuleList(
            CIB(self.c, self.c, shortcut, e=1.0, large_kernel=large_kernel)
            for _ in range(n)
        )

class SPPF(nn.Module):
    def __init__(self, c1, c2, k=5):
        super().__init__()
        if k < 1 or k % 2 == 0:
            raise ValueError("SPPF requires a positive odd kernel size")
        c_ = c1 // 2
        if c_ < 1:
            raise ValueError("SPPF requires c1 >= 2")
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(4 * c_, c2, 1, 1)
        self.m = nn.MaxPool2d(k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.cv2(torch.cat((x, y1, y2, y3), 1))

class DFL(nn.Module):
    def __init__(self, c1=16):
        super().__init__()
        if c1 < 1:
            raise ValueError("DFL requires c1 >= 1")
        self.c1 = c1
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        with torch.no_grad():
            self.conv.weight.copy_(
                torch.arange(c1, dtype=self.conv.weight.dtype).view(1, c1, 1, 1)
            )

    def forward(self, x):
        b, c, a = x.shape
        if c != 4 * self.c1:
            raise ValueError(f"Expected {4 * self.c1} channels, got {c}")
        x = x.reshape(b, 4, self.c1, a).transpose(1, 2)
        return self.conv(x.softmax(1)).reshape(b, 4, a)

class Attention(nn.Module):
    def __init__(self, dim, num_heads=4, mlp_ratio=2.0, layer_scale=1e-2):
        super().__init__()
        if dim < 1 or num_heads < 1 or dim % num_heads:
            raise ValueError("dim must be positive and divisible by num_heads > 0")
        hidden_dim = int(dim * mlp_ratio)
        if hidden_dim < 1:
            raise ValueError("int(dim * mlp_ratio) must be positive")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Conv2d(dim, 3 * dim, 1, bias=False)
        self.proj = Conv(dim, dim, 1, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)
        self.ffn = nn.Sequential(
            Conv(dim, hidden_dim, 1, 1),
            Conv(hidden_dim, dim, 1, 1, act=False),
        )
        self.gamma1 = nn.Parameter(layer_scale * torch.ones(dim))
        self.gamma2 = nn.Parameter(layer_scale * torch.ones(dim))

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(x).reshape(
            B, 3, self.num_heads, self.head_dim, H * W
        )
        q, k, v = qkv.permute(1, 0, 2, 4, 3).contiguous().unbind(0)
        v_spatial = v.transpose(-2, -1).reshape(B, C, H, W)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        out = out.transpose(-2, -1).reshape(B, C, H, W)
        out = self.proj(out + self.pe(v_spatial))

        # Avoid promoting low-precision activations through FP32 LayerScale.
        x = x + self.gamma1.to(out.dtype).view(1, -1, 1, 1) * out
        out = self.ffn(x)
        return x + self.gamma2.to(out.dtype).view(1, -1, 1, 1) * out

class C2fPSA(nn.Module):
    def __init__(self, c1, c2, n=1, e=0.5, num_heads=4):
        super().__init__()
        if c1 != c2:
            raise ValueError("C2fPSA requires c1 == c2")
        self.c = int(c2 * e)
        if self.c < 1 or n < 0:
            raise ValueError("int(c2 * e) must be positive and n nonnegative")
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1, 1)
        self.m = nn.Sequential(
            *(Attention(self.c, num_heads=num_heads) for _ in range(n))
        )

    def forward(self, x):
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((a, self.m(b)), 1))

class SCDown(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, p=None, d=1, act=True):
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1, act=act)
        self.cv2 = Conv(c2, c2, k, s, p=p, g=c2, d=d, act=False)

    def forward(self, x):
        return self.cv2(self.cv1(x))