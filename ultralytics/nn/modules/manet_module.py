# ultralytics/nn/modules/manet_module.py

import torch
import torch.nn as nn
from .conv import Conv


class DSConv(nn.Module):
    """
    Depthwise Separable Convolution:
    DWConv + PWConv，用于轻量化空间特征提取。
    """
    def __init__(self, c1, c2, k=3, s=1):
        super().__init__()
        self.dw = Conv(c1, c1, k, s, g=c1)
        self.pw = Conv(c1, c2, 1, 1)

    def forward(self, x):
        return self.pw(self.dw(x))


class LiteMANetBlock(nn.Module):
    """
    轻量 ConvNeck-like block:
    用残差形式增强局部特征表达。
    """
    def __init__(self, c, k=3, shortcut=True):
        super().__init__()
        self.cv1 = Conv(c, c, k, 1)
        self.cv2 = Conv(c, c, k, 1)
        self.shortcut = shortcut

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.shortcut else y


class LiteMANet(nn.Module):
    """
    Lite-MANet for YOLOv8 backbone.

    Args:
        c1: input channels
        c2: output channels
        n: number of LiteMANetBlock
        k: kernel size, recommend k=5 for P3/P4
        shortcut: whether to use residual connection
        e: hidden channel ratio
    """
    def __init__(self, c1, c2, n=1, k=5, shortcut=True, e=0.5):
        super().__init__()

        c_ = max(int(c2 * e), 8)

        # 输入通道调整，生成两组 hidden features
        self.cv_in = Conv(c1, 2 * c_, 1, 1)

        # branch 1: 1x1 bypass channel recalibration
        self.branch_1 = Conv(c_, c_, 1, 1)

        # branch 2: depthwise separable convolution
        self.branch_2 = DSConv(c_, c_, k=k, s=1)

        # branch 3: lightweight residual aggregation branch
        self.branch_3 = nn.Sequential(
            *[LiteMANetBlock(c_, k=k, shortcut=shortcut) for _ in range(n)]
        )

        # 三分支融合
        self.cv_out = Conv(3 * c_, c2, 1, 1)

    def forward(self, x):
        x = self.cv_in(x)
        x1, x2 = x.chunk(2, dim=1)

        y1 = self.branch_1(x1)
        y2 = self.branch_2(x1)
        y3 = self.branch_3(x2)

        return self.cv_out(torch.cat((y1, y2, y3), dim=1))