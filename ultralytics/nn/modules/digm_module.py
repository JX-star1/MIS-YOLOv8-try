# ultralytics/nn/modules/digm_module.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv


class PFIM(nn.Module):
    """
    Pixels Feature Information Modeling.
    输入 P2，输出信息图 sigma_map 和信息熵损失 lie_loss。
    """

    def __init__(self, c):
        super().__init__()
        self.param_net = nn.Sequential(
            Conv(c, c, 3, 1),
            Conv(c, c, 3, 1),
            nn.Conv2d(c, 2 * c, kernel_size=1)
        )

    def forward(self, x):
        params = self.param_net(x)
        mu, log_scale = torch.chunk(params, 2, dim=1)

        scale = F.softplus(log_scale) + 1e-6

        if self.training:
            noise = torch.empty_like(x).uniform_(-0.5, 0.5)
            y_hat = x + noise
        else:
            y_hat = torch.round(x)

        upper = (y_hat + 0.5 - mu) / scale
        lower = (y_hat - 0.5 - mu) / scale

        cdf_upper = 0.5 * (1.0 + torch.erf(upper / math.sqrt(2.0)))
        cdf_lower = 0.5 * (1.0 + torch.erf(lower / math.sqrt(2.0)))

        likelihood = (cdf_upper - cdf_lower).clamp(min=1e-9)
        bits = -torch.log2(likelihood)

        lie_loss = bits.mean()

        # 用 scale 近似信息量图，压缩成单通道注意力图
        sigma_map = scale.mean(dim=1, keepdim=True)
        sigma_map = torch.sigmoid(sigma_map)

        return sigma_map, lie_loss


class PGDP(nn.Module):
    """
    Position Gaussian Distribution Prediction.
    输入 P2/P3/P4 和 sigma，预测 Mpd2/Mpd3/Mpd4。
    """

    def __init__(self, c2, c3, c4, c_mid=None):
        super().__init__()

        if c_mid is None:
            c_mid = max(c2, 32)

        self.p2_proj = Conv(c2, c_mid, 1, 1)
        self.p3_proj = Conv(c3, c_mid, 1, 1)
        self.p4_proj = Conv(c4, c_mid, 1, 1)

        self.fuse = nn.Sequential(
            Conv(c_mid, c_mid, 3, 1),
            Conv(c_mid, c_mid, 3, 1)
        )

        self.pred2 = nn.Conv2d(c_mid, 1, kernel_size=1)
        self.pred3 = nn.Conv2d(c_mid, 1, kernel_size=1)
        self.pred4 = nn.Conv2d(c_mid, 1, kernel_size=1)

    def forward(self, p2, p3, p4, sigma):
        h2, w2 = p2.shape[-2:]

        sigma_p3 = F.interpolate(
            sigma, size=p3.shape[-2:], mode="bilinear", align_corners=False
        )
        sigma_p4 = F.interpolate(
            sigma, size=p4.shape[-2:], mode="bilinear", align_corners=False
        )

        # sigma 是单通道，可自动 broadcast 到多通道特征
        x2 = self.p2_proj(p2 + sigma)
        x3 = self.p3_proj(p3 + sigma_p3)
        x4 = self.p4_proj(p4 + sigma_p4)

        x3 = F.interpolate(x3, size=(h2, w2), mode="bilinear", align_corners=False)
        x4 = F.interpolate(x4, size=(h2, w2), mode="bilinear", align_corners=False)

        x = self.fuse(x2 + x3 + x4)

        mpd2 = torch.sigmoid(self.pred2(x))
        mpd3 = torch.sigmoid(self.pred3(x))
        mpd4 = torch.sigmoid(self.pred4(x))

        return mpd2, mpd3, mpd4


class DIGM(nn.Module):
    """
    Density-aware Information-Gaussian Modulation.

    输入:
        [P2, P3, P4]

    输出:
        P2'

    Detect 使用:
        Detect(P2', P3, P4)
    """

    def __init__(self, c2, c3, c4):
        super().__init__()

        self.pfim = PFIM(c2)
        self.pgdp = PGDP(c2, c3, c4)

        self.fuse = Conv(c2, c2, 1, 1)

        # 残差门控，初始化为 0，训练初期等价于原 P2
        self.gamma = nn.Parameter(torch.zeros(1))

        # 供 loss.py 读取辅助输出
        self.aux_outputs = {}

    def forward(self, x):
        p2, p3, p4 = x

        sigma, lie_loss = self.pfim(p2)
        mpd2, mpd3, mpd4 = self.pgdp(p2, p3, p4, sigma)

        # 信息图与高斯图交互调制
        sigma_guided = sigma * (1.0 + mpd2)

        p2_info = p2 * (1.0 + sigma_guided)
        p2_gauss = p2 * (1.0 + mpd2)

        p2_enh = self.fuse(p2_info + p2_gauss)

        p2_out = p2 + self.gamma * p2_enh

        self.aux_outputs = {
            "sigma": sigma,
            "mpd2": mpd2,
            "mpd3": mpd3,
            "mpd4": mpd4,
            "lie_loss": lie_loss,
        }

        return p2_out