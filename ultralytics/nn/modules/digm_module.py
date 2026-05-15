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

        # 残差门控
        self.gamma = nn.Parameter(torch.zeros(0.1))

        # 默认不收集 aux，避免初始化 dummy forward 后模型 deepcopy 报错
        self.save_aux = False
        self.aux_outputs = None

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

        # 只有训练 loss 显式打开 save_aux 时才保存带梯度的辅助输出
        if self.save_aux:
            self.aux_outputs = {
                "sigma": sigma,
                "mpd2": mpd2,
                "mpd3": mpd3,
                "mpd4": mpd4,
                "lie_loss": lie_loss,
            }
        else:
            self.aux_outputs = None

        return p2_out

@torch.no_grad()
def build_density_gaussian_map(batch, pred_map, beta=0.2, radius=16.0):
    """
    根据 YOLOv8 batch 标注生成密度感知高斯 GT 图。

    Args:
        batch: YOLOv8 training batch.
               batch["bboxes"] 通常是 normalized xywh.
               batch["batch_idx"] 表示每个框属于 batch 中第几张图.
        pred_map: [B, 1, H, W], 例如 mpd2.
        beta: 密度调节系数.
        radius: 在 P2 特征图尺度下统计邻居的半径.

    Returns:
        mgt: [B, 1, H, W]
    """
    device = pred_map.device
    b, _, h, w = pred_map.shape

    mgt = torch.zeros((b, 1, h, w), device=device)

    if "bboxes" not in batch or batch["bboxes"].numel() == 0:
        return mgt

    boxes = batch["bboxes"].to(device)
    batch_idx = batch["batch_idx"].to(device).long().view(-1)

    # YOLOv8 bboxes: normalized xywh
    cx = boxes[:, 0] * w
    cy = boxes[:, 1] * h
    bw = boxes[:, 2] * w
    bh = boxes[:, 3] * h

    img_h, img_w = batch["img"].shape[-2:]
    stride_x = img_w / float(w)
    stride_y = img_h / float(h)

    yy, xx = torch.meshgrid(
        torch.arange(h, device=device),
        torch.arange(w, device=device),
        indexing="ij"
    )
    xx = xx.float()
    yy = yy.float()

    for img_i in range(b):
        inds = torch.where(batch_idx == img_i)[0]
        if inds.numel() == 0:
            continue

        centers = torch.stack([cx[inds], cy[inds]], dim=1)

        for local_j, idx in enumerate(inds):
            x0 = cx[idx]
            y0 = cy[idx]
            ww = bw[idx].clamp(min=1.0)
            hh = bh[idx].clamp(min=1.0)

            # 换回原图尺度估计目标大小
            obj_w = ww * stride_x
            obj_h = hh * stride_y
            obj_size = torch.sqrt(obj_w * obj_h)

            # 参考 tiny object 划分：very tiny / tiny / small / general
            if obj_size < 8:
                alpha = 4.0
            elif obj_size < 16:
                alpha = 6.0
            elif obj_size < 32:
                alpha = 8.0
            else:
                alpha = 10.0

            # 密度感知：邻居越多，高斯越尖锐，减少密集目标响应粘连
            dist = torch.sqrt(((centers - centers[local_j]) ** 2).sum(dim=1))
            n_neighbor = (dist < radius).sum().float() - 1.0
            density = 1.0 + beta * torch.log1p(n_neighbor.clamp(min=0.0))

            alpha = alpha * density

            sx = (ww / alpha).clamp(min=0.5)
            sy = (hh / alpha).clamp(min=0.5)

            gaussian = torch.exp(
                -0.5 * (((xx - x0) / sx) ** 2 + ((yy - y0) / sy) ** 2)
            )

            # 用 maximum 避免密集目标简单相加导致过强响应
            mgt[img_i, 0] = torch.maximum(mgt[img_i, 0], gaussian)

    return mgt.clamp(0.0, 1.0)


def weighted_gaussian_mse(pred, target, fg_thr=0.05):
    """
    前景区域高权重，背景区域低权重。
    """
    weight = torch.where(
        target > fg_thr,
        torch.tensor(10.0, device=target.device, dtype=target.dtype),
        torch.tensor(0.1, device=target.device, dtype=target.dtype),
    )
    return (weight * (pred - target) ** 2).mean()


def digm_auxiliary_loss(model, batch, lambda_ie=0.001, lambda_gauss=0.25, supervise_all=False):
    """
    DIGM 辅助损失:
        L_aux = lambda_ie * L_IE + lambda_gauss * L_gauss

    supervise_all=False:
        只监督 mpd2，第一版更稳。

    supervise_all=True:
        同时监督 mpd2/mpd3/mpd4。
    """
    digm_modules = [m for m in model.modules() if isinstance(m, DIGM)]
    if len(digm_modules) == 0:
        return None

    digm = digm_modules[0]

    if not hasattr(digm, "aux_outputs") or digm.aux_outputs is None:
        return None

    aux = digm.aux_outputs

    lie_loss = aux["lie_loss"]
    mpd2 = aux["mpd2"]

    mgt2 = build_density_gaussian_map(batch, mpd2)

    if supervise_all:
        mpd3 = aux["mpd3"]
        mpd4 = aux["mpd4"]

        mgt3 = F.interpolate(mgt2, size=mpd3.shape[-2:], mode="nearest")
        mgt4 = F.interpolate(mgt2, size=mpd4.shape[-2:], mode="nearest")

        l_gauss = (
            weighted_gaussian_mse(mpd2, mgt2) +
            weighted_gaussian_mse(mpd3, mgt3) +
            weighted_gaussian_mse(mpd4, mgt4)
        ) / 3.0
    else:
        l_gauss = weighted_gaussian_mse(mpd2, mgt2)

    return lambda_ie * lie_loss + lambda_gauss * l_gauss