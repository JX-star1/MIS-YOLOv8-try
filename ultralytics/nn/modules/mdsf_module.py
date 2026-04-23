import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv


# -----------------------------
# Depthwise Separable Conv
# -----------------------------
class DWConv(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, p=None):
        super().__init__()
        if p is None:
            p = k // 2

        self.dw = nn.Conv2d(c1, c1, k, s, p, groups=c1, bias=False)
        self.pw = nn.Conv2d(c1, c2, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))


# -----------------------------
# Depthwise Dilated Conv
# -----------------------------
class DWDConv(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, d=1):
        super().__init__()
        p = ((k - 1) // 2) * d

        self.dw = nn.Conv2d(c1, c1, k, s, p, dilation=d, groups=c1, bias=False)
        self.pw = nn.Conv2d(c1, c2, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))


# -----------------------------
# MDSF Block (论文核心模块)
# -----------------------------
class MDSF4(nn.Module):
    """
    Multi-scale Depthwise Separable Fusion (MDSF)

    特点：
    - 多分支不同感受野
    - 深度可分离卷积（轻量）
    - 空洞卷积（扩大感受野）
    - 可选残差（稳定训练）
    """

    def __init__(self, c1, c2, shortcut=True, e=0.5):
        super().__init__()
        self.shortcut = shortcut

        # 中间通道（类似C2f思想，降计算）
        c_ = int(c2 * e)

        # -------------------
        # Branch 1 (small RF)
        # -------------------
        self.b1 = nn.Sequential(
            DWConv(c1, c_, k=3),
            DWDConv(c_, c_, k=3, d=1)
        )

        # -------------------
        # Branch 2 (medium RF)
        # -------------------
        self.b2 = nn.Sequential(
            DWConv(c1, c_, k=5),
            DWDConv(c_, c_, k=5, d=2)
        )

        # -------------------
        # Branch 3 (large RF)
        # -------------------
        self.b3 = nn.Sequential(
            DWConv(c1, c_, k=7),
            DWDConv(c_, c_, k=7, d=3)
        )

        # 融合
        self.fuse = Conv(c_ * 3, c2, k=1)

        # shortcut 对齐
        self.short = Conv(c1, c2, k=1, act=False) if c1 != c2 else nn.Identity()

    def forward(self, x):
        identity = self.short(x)

        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)

        out = torch.cat((y1, y2, y3), dim=1)
        out = self.fuse(out)

        if self.shortcut:
            out = out + identity

        return out