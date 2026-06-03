# ultralytics/nn/modules/manet_module.py

import torch
import torch.nn as nn
from .conv import Conv


class DSConv(nn.Module):
    """Depthwise Separable Convolution: DWConv + PWConv."""
    def __init__(self, c1, c2, k=3, s=1):
        super().__init__()
        self.dw = Conv(c1, c1, k, s, g=c1)
        self.pw = Conv(c1, c2, 1, 1)

    def forward(self, x):
        return self.pw(self.dw(x))


class ConvNeck(nn.Module):
    """
    原文 MANet 中的 ConvNeck 分支：
    两个 k×k Conv，外部做 residual。
    """
    def __init__(self, c, k=3):
        super().__init__()
        self.cv1 = Conv(c, c, k, 1)
        self.cv2 = Conv(c, c, k, 1)

    def forward(self, x):
        return self.cv2(self.cv1(x))


class MANet(nn.Module):
    """
    MANet according to Hyper-YOLO paper.

    Args:
        c1: input channels
        c2: output channels
        n: number of ConvNeck residual blocks
        k: kernel size
        shortcut: whether to use residual connection in ConvNeck branch
        e: hidden ratio. e=0.5 means c2 = 2c, matching paper notation.
    """
    def __init__(self, c1, c2, n=1, k=3, shortcut=True, e=0.5):
        super().__init__()

        c = max(int(c2 * e), 8)

        # Xmid = Conv1(Xin), channel = 2c
        self.cv_in = Conv(c1, 2 * c, 1, 1)

        # X1 = Conv2(Xmid), 1×1 bypass branch
        self.branch_1 = Conv(2 * c, c, 1, 1)

        # X2 = DSConv(Conv3(Xmid))
        # 这里先 1×1 扩到 4c，再 DWConv + PWConv 压回 c
        self.branch_2 = nn.Sequential(
            Conv(2 * c, 4 * c, 1, 1),
            DSConv(4 * c, c, k=k, s=1)
        )

        # X5...X4+n = ConvNeck residual branch
        self.blocks = nn.ModuleList([ConvNeck(c, k=k) for _ in range(n)])
        self.shortcut = shortcut

        # concat: X1, X2, X3, X4, X5...X4+n
        # 共 4+n 个 c 通道特征
        self.cv_out = Conv((4 + n) * c, c2, 1, 1)

    def forward(self, x):
        x_mid = self.cv_in(x)

        x1 = self.branch_1(x_mid)
        x2 = self.branch_2(x_mid)

        x3, x4 = x_mid.chunk(2, dim=1)

        outs = [x1, x2, x3, x4]

        y = x4
        for block in self.blocks:
            y = block(y) + y if self.shortcut else block(y)
            outs.append(y)

        return self.cv_out(torch.cat(outs, dim=1))