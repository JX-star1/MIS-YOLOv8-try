import torch
import torch.nn as nn

from .conv import Conv


class PGDHeatGate(nn.Module):
    """
    PGDHeatGate:
    Predict a class-agnostic Gaussian heatmap and use it to enhance small-object features.

    Output:
        gated feature map with the same spatial size as input.
    """

    def __init__(self, c1, c2, stride=4, gate=0.5, loss_weight=1.0):
        super().__init__()
        self.stride_level = int(stride)
        self.gate = float(gate)
        self.loss_weight = float(loss_weight)

        self.proj = Conv(c1, c2, 1, 1) if c1 != c2 else nn.Identity()

        hidden = max(c2 // 2, 8)
        self.heat_head = nn.Sequential(
            Conv(c2, c2, 3, 1),
            Conv(c2, hidden, 3, 1),
            nn.Conv2d(hidden, 1, kernel_size=1, stride=1, padding=0),
        )

        self.last_heat = None
        self.save_heat = False

    def forward(self, x):
        x = self.proj(x)
        heat = torch.sigmoid(self.heat_head(x))
        self.last_heat = heat

        # gate=0 时只保留辅助监督，不改变特征
        return x * (1.0 + self.gate * heat)


def build_pgd_target(batch, pred, img_size, small_classes=(0, 1, 2, 6, 7, 9), max_obj_size=64):
    """
    Build class-agnostic Gaussian heatmap target.

    VisDrone class order usually:
    0 pedestrian, 1 people, 2 bicycle, 3 car, 4 van,
    5 truck, 6 tricycle, 7 awning-tricycle, 8 bus, 9 motor.
    """

    bsz, _, hf, wf = pred.shape
    device = pred.device
    dtype = pred.dtype
    img_h, img_w = img_size

    target = torch.zeros((bsz, 1, hf, wf), device=device, dtype=dtype)
    weight = torch.full_like(target, 0.1)

    if "bboxes" not in batch or batch["bboxes"].numel() == 0:
        return target, weight

    bboxes = batch["bboxes"].to(device).float()
    batch_idx = batch["batch_idx"].to(device).long().view(-1)
    cls = batch["cls"].to(device).long().view(-1)

    # Ultralytics 通常为归一化 xywh
    if bboxes.max() <= 2.0:
        cx = bboxes[:, 0] * img_w
        cy = bboxes[:, 1] * img_h
        bw = bboxes[:, 2] * img_w
        bh = bboxes[:, 3] * img_h
    else:
        # 如果你的数据流已经是像素 xywh，则走这里
        cx = bboxes[:, 0]
        cy = bboxes[:, 1]
        bw = bboxes[:, 2]
        bh = bboxes[:, 3]

    sx = img_w / wf
    sy = img_h / hf

    xs = torch.arange(wf, device=device, dtype=dtype).view(1, wf)
    ys = torch.arange(hf, device=device, dtype=dtype).view(hf, 1)

    for i in range(bboxes.shape[0]):
        c = int(cls[i].item())
        b = int(batch_idx[i].item())

        if b < 0 or b >= bsz:
            continue

        area = float((bw[i] * bh[i]).item())

        # 只强化小目标类，或者尺寸很小的大类目标
        if not (c in small_classes or area <= max_obj_size * max_obj_size):
            continue

        if bw[i] <= 1 or bh[i] <= 1:
            continue

        gx = cx[i] / sx
        gy = cy[i] / sy
        gw = bw[i] / sx
        gh = bh[i] / sy

        max_side = max(float(bw[i].item()), float(bh[i].item()))

        if max_side < 16:
            alpha = 2.0
        elif max_side < 32:
            alpha = 3.0
        elif max_side < 64:
            alpha = 4.0
        else:
            alpha = 6.0

        sigx = torch.clamp(gw / alpha, min=1.0)
        sigy = torch.clamp(gh / alpha, min=1.0)

        g = torch.exp(-0.5 * (((xs - gx) / sigx) ** 2 + ((ys - gy) / sigy) ** 2))

        target[b, 0] = torch.maximum(target[b, 0], g)

    target = target.clamp(0, 1)
    weight = torch.where(target > 0.05, torch.full_like(weight, 5.0), weight)

    return target, weight


def pgd_heatmap_loss(model, batch, lambda_heat=0.05):
    """
    Compute auxiliary heatmap loss for all PGDHeatGate modules.
    """

    if "img" not in batch:
        return None

    img_h, img_w = batch["img"].shape[-2:]
    losses = []

    for m in model.modules():
        if isinstance(m, PGDHeatGate) and m.last_heat is not None:
            pred = m.last_heat
            target, weight = build_pgd_target(batch, pred, (img_h, img_w))
            loss = (weight * (pred - target).pow(2)).mean()
            losses.append(loss * m.loss_weight)

    if not losses:
        return None

    return torch.stack(losses).sum() * lambda_heat