# ultralytics/nn/modules/spa_module.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv


class SPAFeature(nn.Module):
    """
    Small-object Prior Spatial Pathway Aggregation.

    输入:
        [P2, P3, P4, P5ctx]

    输出:
        level=0 -> Y2
        level=1 -> Y3
        level=2 -> Y4

    YAML 示例:
        - [[18, 21, 24, 9], 1, SPAFeature, [0]]
        - [[18, 21, 24, 9], 1, SPAFeature, [1]]
        - [[18, 21, 24, 9], 1, SPAFeature, [2]]
    """

    def __init__(self, c2, c3, c4, c5, level=0, use_prior=True):
        super().__init__()

        assert level in (0, 1, 2), f"SPAFeature level must be 0, 1, or 2, got {level}"

        self.level = level
        self.use_prior = use_prior

        channels = [c2, c3, c4, c5]
        out_channels = [c2, c3, c4][level]
        self.out_channels = out_channels

        # 将四个尺度统一到当前输出尺度的通道数
        self.proj = nn.ModuleList([
            Conv(ch, out_channels, 1, 1) for ch in channels
        ])

        # 根据拼接后的四尺度特征预测空间动态权重
        self.weight_net = nn.Sequential(
            Conv(out_channels * 4, out_channels, 1, 1),
            nn.Conv2d(out_channels, 4, kernel_size=1)
        )

        # 融合后的轻量调整
        self.fuse = Conv(out_channels, out_channels, 3, 1)

        # 残差门控，初始化为 0，训练初期接近原 YOLOv8-P2-noP5
        self.gamma = nn.Parameter(torch.zeros(1))

        # 小目标先验偏置，对应 [P2, P3, P4, P5ctx]
        if level == 0:
            prior = [2.0, 1.0, 0.0, -1.0]      # Y2 更偏 P2
        elif level == 1:
            prior = [1.0, 1.5, 0.5, -0.5]      # Y3 更偏 P3
        else:
            prior = [0.5, 1.0, 1.5, 0.0]       # Y4 更偏 P4

        self.register_buffer("prior", torch.tensor(prior).view(1, 4, 1, 1).float())

    def _project_and_resize(self, x, proj, target_size):
        """
        为了节省计算:
        - 如果输入分辨率大于目标分辨率，先 resize 再 1x1 conv；
        - 如果输入分辨率小于目标分辨率，先 1x1 conv 再 upsample。
        """
        h, w = x.shape[-2:]
        th, tw = target_size
        area = h * w
        target_area = th * tw

        if area > target_area:
            x = F.interpolate(x, size=target_size, mode="nearest")
            x = proj(x)
        else:
            x = proj(x)
            if x.shape[-2:] != target_size:
                x = F.interpolate(x, size=target_size, mode="nearest")

        return x

    def forward(self, x):
        # x = [P2, P3, P4, P5ctx]
        p2, p3, p4, p5 = x
        feats = [p2, p3, p4, p5]

        target = feats[self.level]
        target_size = target.shape[-2:]

        aligned = [
            self._project_and_resize(feats[i], self.proj[i], target_size)
            for i in range(4)
        ]

        # 预测空间动态权重
        weight_input = torch.cat(aligned, dim=1)
        logits = self.weight_net(weight_input)

        if self.use_prior:
            logits = logits + self.prior.to(dtype=logits.dtype, device=logits.device)

        weights = torch.softmax(logits, dim=1)

        # 加权融合
        fused = (
            weights[:, 0:1] * aligned[0] +
            weights[:, 1:2] * aligned[1] +
            weights[:, 2:3] * aligned[2] +
            weights[:, 3:4] * aligned[3]
        )

        fused = self.fuse(fused)

        # 残差门控
        out = target + self.gamma * fused

        return out