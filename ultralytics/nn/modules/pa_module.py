import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv


class _PAAlignBase(nn.Module):
    """
    Base alignment module for APA ablation.

    Input:
        x = [P2, P3, P4]

    target=0 -> output scale P2
    target=1 -> output scale P3
    target=2 -> output scale P4

    learn_down=True:
        high-resolution -> low-resolution uses stride-2 Conv.
        This is the alignment strategy used by APA.

    learn_down=False:
        high-resolution -> low-resolution uses interpolation after channel alignment.
        This is used for ASFF-like comparison to avoid making it identical to APA.
    """

    def __init__(self, ch, target=0, learn_down=True):
        super().__init__()
        self.ch = ch
        self.target = int(target)
        self.nl = len(ch)
        self.c2 = ch[self.target]
        self.learn_down = learn_down

        self.align = nn.ModuleList()

        for i, c in enumerate(ch):
            ops = []

            # Source feature has higher resolution than target.
            # Example: P2 -> P3, P2 -> P4, P3 -> P4.
            if i < self.target:
                if self.learn_down:
                    in_c = c
                    for _ in range(self.target - i):
                        out_c = self.c2
                        ops.append(Conv(in_c, out_c, 3, 2))
                        in_c = out_c
                else:
                    if c != self.c2:
                        ops.append(Conv(c, self.c2, 1, 1))
                    else:
                        ops.append(nn.Identity())

            # Same resolution.
            elif i == self.target:
                if c != self.c2:
                    ops.append(Conv(c, self.c2, 1, 1))
                else:
                    ops.append(nn.Identity())

            # Source feature has lower resolution than target.
            # Example: P3 -> P2, P4 -> P2, P4 -> P3.
            else:
                if c != self.c2:
                    ops.append(Conv(c, self.c2, 1, 1))
                else:
                    ops.append(nn.Identity())

            self.align.append(nn.Sequential(*ops))

    def _align_feats(self, x):
        assert isinstance(x, (list, tuple)), "PA module expects a list of feature maps."

        target_size = x[self.target].shape[-2:]
        feats = []

        for i, xi in enumerate(x):
            xi = self.align[i](xi)

            if xi.shape[-2:] != target_size:
                xi = F.interpolate(xi, size=target_size, mode="nearest")

            feats.append(xi)

        return feats


class PAAdd(_PAAlignBase):
    """
    Add Fusion:
        Align P2/P3/P4 to target scale, then directly add them.
    """

    def __init__(self, ch, target=0):
        super().__init__(ch, target=target, learn_down=True)
        self.out = Conv(self.c2, self.c2, 3, 1)

    def forward(self, x):
        feats = self._align_feats(x)

        y = 0
        for feat in feats:
            y = y + feat

        return self.out(y)


class PAConcat(_PAAlignBase):
    """
    Concat Fusion:
        Align P2/P3/P4 to target scale, concatenate them, then use Conv.
    """

    def __init__(self, ch, target=0):
        super().__init__(ch, target=target, learn_down=True)
        self.out = Conv(self.c2 * self.nl, self.c2, 3, 1)

    def forward(self, x):
        feats = self._align_feats(x)
        return self.out(torch.cat(feats, dim=1))


class PABiFPN(_PAAlignBase):
    """
    BiFPN-like Fusion:
        Align P2/P3/P4 to target scale, then fuse them with learnable scalar weights.
        The weights are global for each source scale, not spatial-adaptive.
    """

    def __init__(self, ch, target=0, eps=1e-4):
        super().__init__(ch, target=target, learn_down=True)
        self.eps = eps
        self.w = nn.Parameter(torch.ones(self.nl, dtype=torch.float32))
        self.out = Conv(self.c2, self.c2, 3, 1)

    def forward(self, x):
        feats = self._align_feats(x)

        w = F.relu(self.w)
        w = w / (w.sum() + self.eps)

        y = 0
        for i, feat in enumerate(feats):
            y = y + w[i] * feat

        return self.out(y)


class PAASFF(_PAAlignBase):
    """
    ASFF-like Fusion:
        Align P2/P3/P4 to target scale, then predict pixel-wise source weights.

    Difference from APA:
        1) high-resolution -> low-resolution uses non-learnable resize after channel alignment;
        2) weight prediction uses compressed features, similar to common ASFF-style lightweight weighting.
    """

    def __init__(self, ch, target=0, compress_ratio=8):
        super().__init__(ch, target=target, learn_down=False)

        c_ = max(self.c2 // compress_ratio, 8)

        self.weight_convs = nn.ModuleList(
            [Conv(self.c2, c_, 1, 1) for _ in range(self.nl)]
        )
        self.weight = nn.Conv2d(c_ * self.nl, self.nl, kernel_size=1, stride=1, padding=0)
        self.out = Conv(self.c2, self.c2, 3, 1)

    def forward(self, x):
        feats = self._align_feats(x)

        weight_feats = []
        for i, feat in enumerate(feats):
            weight_feats.append(self.weight_convs[i](feat))

        w = self.weight(torch.cat(weight_feats, dim=1))
        w = torch.softmax(w, dim=1)

        y = 0
        for i, feat in enumerate(feats):
            y = y + feat * w[:, i:i + 1]

        return self.out(y)


class PAFeature(_PAAlignBase):
    """
    APA / final method:
        Align P2/P3/P4 to target scale.
        High-resolution -> low-resolution uses learnable stride-2 Conv.
        Then predict pixel-wise source weights from full concatenated multi-scale features.
    """

    def __init__(self, ch, target=0):
        super().__init__(ch, target=target, learn_down=True)

        self.weight = nn.Sequential(
            Conv(self.c2 * self.nl, self.c2, 1, 1),
            nn.Conv2d(self.c2, self.nl, kernel_size=1, stride=1, padding=0),
        )

        self.out = Conv(self.c2, self.c2, 3, 1)

        # add for visualization
        self.last_weight = None
        self.save_weight = False

    def forward(self, x):
        feats = self._align_feats(x)

        w = self.weight(torch.cat(feats, dim=1))
        w = torch.softmax(w, dim=1)

        if self.save_weight:
            self.last_weight = w.detach()
        else:
            self.last_weight = None

        y = 0
        for i, feat in enumerate(feats):
            y = y + feat * w[:, i:i + 1]

        return self.out(y)