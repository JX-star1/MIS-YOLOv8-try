import torch
import torch.nn as nn

from .conv import Conv, DWConv
from .mdsf_module import MDSF4


class DFF(nn.Module):
    """
    DFF: Deep Feature Fusion module.

    Structure:
    input -> 1x1 Conv -> DWConv -> MaxPool x3 -> Concat -> 1x1 Conv -> MDSF4
          \_______________________________________________________________/
                               residual 1x1 Conv
    """

    def __init__(self, c1, c2, k=5, shortcut=True):
        super().__init__()

        c_ = c1 // 2

        self.cv1 = Conv(c1, c_, 1, 1)
        self.dw = DWConv(c_, c_, 3, 1)

        self.pool = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.mdsf = MDSF4(c2, c2, shortcut)

        self.res = Conv(c1, c2, 1, 1) if c1 != c2 else nn.Identity()

    def forward(self, x):
        residual = self.res(x)

        x = self.cv1(x)
        x = self.dw(x)

        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)

        y = torch.cat((x, y1, y2, y3), dim=1)
        y = self.cv2(y)
        y = self.mdsf(y)

        return y + residual