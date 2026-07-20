# ultralytics/nn/modules/ls_module.py

import torch
import torch.nn as nn
from .conv import Conv
from .block import C2f, SPPF


class LSConvLite(nn.Module):
    """
    Deployment-friendly LSConv-Lite:
    1x1 Conv -> 7x7 DWConv -> 3x3 DWConv -> 1x1 Conv
    用标准算子近似 LSNet 的 See Large + Focus Small 思想。
    """
    def __init__(self, c1, c2, k_large=7, k_small=3, s=1, e=0.5):
        super().__init__()
        c_ = max(int(c2 * e), 8)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.dw_large = Conv(c_, c_, k_large, s, g=c_)
        self.dw_small = Conv(c_, c_, k_small, 1, g=c_)
        self.cv2 = Conv(c_, c2, 1, 1)
        self.use_res = (c1 == c2 and s == 1)

    def forward(self, x):
        y = self.cv2(self.dw_small(self.dw_large(self.cv1(x))))
        return x + y if self.use_res else y


class LSConvNeck(nn.Module):
    """用于替换 MANet 里的 ConvNeck。"""
    def __init__(self, c, k=5):
        super().__init__()
        self.cv1 = LSConvLite(c, c, k_large=7, k_small=3)
        self.cv2 = Conv(c, c, 1, 1)

    def forward(self, x):
        return self.cv2(self.cv1(x))


class LSMANet(nn.Module):
    """
    与你现有 MANet 参数形式保持一致：
    LSMANet(c1, c2, n=1, k=3, shortcut=True, e=0.5)
    """
    def __init__(self, c1, c2, n=1, k=3, shortcut=True, e=0.5):
        super().__init__()
        c = max(int(c2 * e), 8)
        self.cv_in = Conv(c1, 2 * c, 1, 1)

        self.branch_1 = Conv(2 * c, c, 1, 1)

        self.branch_2 = nn.Sequential(
            Conv(2 * c, 4 * c, 1, 1),
            LSConvLite(4 * c, c, k_large=7, k_small=3)
        )

        self.blocks = nn.ModuleList([LSConvNeck(c, k=k) for _ in range(n)])
        self.shortcut = shortcut
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


class LSC2f(C2f):
    """
    简单继承 C2f，第一版可以先保持 C2f 结构。
    后续如果要进一步轻量化，再把内部 Bottleneck 替换成 LSConvLite。
    """
    pass


class LSSPPF(SPPF):
    """
    第一版保留 SPPF 逻辑，只是作为独立模块名，方便 YAML 消融。
    后续可进一步替换内部 Conv。
    """
    pass