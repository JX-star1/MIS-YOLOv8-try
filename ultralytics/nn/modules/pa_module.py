import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv


class PAFeature(nn.Module):
    """
    PAFeature: Pathway Aggregation before Detect.

    Input:
        x = [P2, P3, P4]

    Output:
        one fused feature map for target level:
        target=0 -> PA-P2
        target=1 -> PA-P3
        target=2 -> PA-P4
    """

    def __init__(self, ch, target=0):
        super().__init__()

        self.ch = ch
        self.target = target
        self.nl = len(ch)
        self.c2 = ch[target]

        # Align channels of all input levels to target channel.
        self.proj = nn.ModuleList(
            Conv(c, self.c2, 1, 1) if c != self.c2 else nn.Identity()
            for c in ch
        )

        # Learn spatial pathway weights.
        # Input: concat(P2, P3, P4) after resize and channel alignment.
        # Output: [B, 3, H, W], then softmax over pathway dimension.
        self.weight = nn.Sequential(
            Conv(self.c2 * self.nl, self.c2, 1, 1),
            nn.Conv2d(self.c2, self.nl, kernel_size=1, stride=1, padding=0)
        )

        self.out = Conv(self.c2, self.c2, 3, 1)

    def forward(self, x):
        assert isinstance(x, (list, tuple)), "PAFeature expects a list of feature maps."

        target_size = x[self.target].shape[-2:]
        feats = []

        for i, xi in enumerate(x):
            xi = self.proj[i](xi)

            if xi.shape[-2:] != target_size:
                xi = F.interpolate(xi, size=target_size, mode="nearest")

            feats.append(xi)

        # Spatial weights for each pathway.
        w = self.weight(torch.cat(feats, dim=1))
        w = torch.softmax(w, dim=1)

        y = 0
        for i, feat in enumerate(feats):
            y = y + feat * w[:, i:i + 1]

        return self.out(y)