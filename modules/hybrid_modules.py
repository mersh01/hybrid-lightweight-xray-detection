
from __future__ import annotations
import torch
import torch.nn as nn

class DualConv(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, p=None):
        super().__init__()
        if p is None:
            p = k // 2
        self.gconv = nn.Conv2d(c1, c1, k, s, p, groups=c1, bias=False)
        self.bn1 = nn.BatchNorm2d(c1)
        self.pconv = nn.Conv2d(c1, c2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True)
    def forward(self, x):
        return self.act(self.bn2(self.pconv(self.act(self.bn1(self.gconv(x))))))

class SobelConv(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True)
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = sobel_x.T
        with torch.no_grad():
            for i in range(min(c2, c1)):
                self.conv.weight[i, i % c1] = sobel_x if i % 2 == 0 else sobel_y
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class FDDN(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.pool = nn.AvgPool2d(3, stride=1, padding=1)
        self.low_path = DualConv(c1, c2 // 2)
        self.high_path = SobelConv(c1, c2 // 2)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c2, c2 // 4, 1), nn.SiLU(),
            nn.Conv2d(c2 // 4, c2, 1), nn.Sigmoid(),
        )
    def forward(self, x):
        low = self.pool(x)
        high = x - low
        feat = torch.cat([self.low_path(low), self.high_path(high)], dim=1)
        return feat * self.se(feat)

class SSCAM(nn.Module):
    def __init__(self, c1, reduction=16):
        super().__init__()
        mid = max(c1 // reduction, 4)
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, mid, 1, bias=False), nn.ReLU(inplace=True),
            nn.Conv2d(mid, c1, 1, bias=False), nn.Sigmoid(),
        )
        self.sa = nn.Sequential(nn.Conv2d(2, 1, 7, padding=3, bias=False), nn.Sigmoid())
    def forward(self, x):
        out = x * self.ca(x)
        avg = torch.mean(out, dim=1, keepdim=True)
        mx, _ = torch.max(out, dim=1, keepdim=True)
        return out * self.sa(torch.cat([avg, mx], dim=1))

class DAPA_FPN(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fg_branch = nn.Sequential(DualConv(c1, c2), SSCAM(c2))
        self.bg_gate = nn.Sequential(
            nn.Conv2d(c1, c2, 1, bias=False), nn.BatchNorm2d(c2), nn.Sigmoid(),
        )
        self.fuse = DualConv(c2, c2)
    def forward(self, x):
        return self.fuse(self.fg_branch(x) * (1.0 + self.bg_gate(x)))

class HybridBlock(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.block = nn.Sequential(FDDN(c1, c2), SSCAM(c2))
    def forward(self, x):
        return self.block(x)

class DAPABlock(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.block = DAPA_FPN(c1, c2)
    def forward(self, x):
        return self.block(x)
