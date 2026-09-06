# tools/vis_pgd_heatmap.py
import argparse
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch

from ultralytics import YOLO
from ultralytics.nn.modules.pgd_module import PGDHeatGate, build_pgd_target

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize PGD heatmaps and feature responses.")
    parser.add_argument("--weights", default="runs/pgap_yolo_s_200e/weights/best.pt")
    parser.add_argument("--image", default="vis_inputs/dense_all_0000295_02400_d_0000033.jpg")
    parser.add_argument("--label", default="vis_inputs/dense_all_0000295_02400_d_0000033.txt")
    parser.add_argument("--output-dir", default="vis_outputs/pgd")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0", help="Ultralytics device string, for example 0 or cpu.")
    return parser.parse_args()


def letterbox_resize(img, new_size=640):
    """简单起见，这里直接resize；如果你想完全与训练一致，可后续换成ultralytics的letterbox。"""
    return cv2.resize(img, (new_size, new_size), interpolation=cv2.INTER_LINEAR)


def normalize_map(x):
    x = x.astype(np.float32)
    x = x - x.min()
    if x.max() > 1e-6:
        x = x / x.max()
    return x


def overlay_heatmap(img_bgr, heat):
    heat = normalize_map(heat)
    heat_u8 = np.uint8(255 * heat)
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_bgr, 0.55, heat_color, 0.45, 0)
    return overlay


def load_yolo_label(label_path):
    if not os.path.exists(label_path):
        return np.zeros((0, 5), dtype=np.float32)

    rows = []
    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if line == "":
                continue
            vals = list(map(float, line.split()))
            rows.append(vals)
    if len(rows) == 0:
        return np.zeros((0, 5), dtype=np.float32)
    return np.array(rows, dtype=np.float32)  # [N,5], cls x y w h


def feature_response_map(feat):
    # feat: [1, C, H, W]
    fmap = feat.abs().mean(dim=1, keepdim=False)[0].cpu().numpy()
    return fmap


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载模型
    model = YOLO(args.weights)
    net = model.model

    # 打开 PGD 缓存
    pgd_modules = []
    for m in net.modules():
        if isinstance(m, PGDHeatGate):
            m.save_heat = True
            pgd_modules.append(m)

    # 跑一次 predict，触发缓存
    _ = model.predict(
        source=args.image,
        imgsz=args.imgsz,
        conf=0.25,
        iou=0.7,
        device=args.device,
        save=False,
        verbose=False
    )

    # 默认取第一个 PGD 模块（P2 分支）
    pgd_p2 = pgd_modules[0]

    pred_heat = pgd_p2.last_heat
    feat_in = pgd_p2.last_feat_in
    feat_out = pgd_p2.last_feat_out

    if pred_heat is None:
        raise RuntimeError("pred_heat is None. Check PGDHeatGate save_heat.")

    # 读原图
    raw_bgr = cv2.imread(args.image)
    raw_bgr = letterbox_resize(raw_bgr, args.imgsz)
    H, W = raw_bgr.shape[:2]
    raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)

    # 构造 batch，生成 GT Gaussian target
    labels = load_yolo_label(args.label)
    img_tensor = torch.from_numpy(raw_rgb.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    img_tensor = img_tensor.to(pred_heat.device)

    if labels.shape[0] > 0:
        cls = torch.from_numpy(labels[:, 0:1]).float().to(pred_heat.device)
        bboxes = torch.from_numpy(labels[:, 1:5]).float().to(pred_heat.device)
        batch_idx = torch.zeros((labels.shape[0],), dtype=torch.long, device=pred_heat.device)
    else:
        cls = torch.zeros((0, 1), dtype=torch.float32, device=pred_heat.device)
        bboxes = torch.zeros((0, 4), dtype=torch.float32, device=pred_heat.device)
        batch_idx = torch.zeros((0,), dtype=torch.long, device=pred_heat.device)

    batch = {
        "img": img_tensor,
        "cls": cls,
        "bboxes": bboxes,
        "batch_idx": batch_idx,
    }

    target, _ = build_pgd_target(batch, pred_heat, (H, W))
    gt_heat = target[0, 0].cpu().numpy()
    pred_heat_np = pred_heat[0, 0].cpu().numpy()

    feat_before = feature_response_map(feat_in)
    feat_after = feature_response_map(feat_out)

    # resize到原图大小
    gt_heat = cv2.resize(gt_heat, (W, H), interpolation=cv2.INTER_LINEAR)
    pred_heat_np = cv2.resize(pred_heat_np, (W, H), interpolation=cv2.INTER_LINEAR)
    feat_before = cv2.resize(feat_before, (W, H), interpolation=cv2.INTER_LINEAR)
    feat_after = cv2.resize(feat_after, (W, H), interpolation=cv2.INTER_LINEAR)

    # overlay
    gt_overlay = cv2.cvtColor(overlay_heatmap(raw_bgr, gt_heat), cv2.COLOR_BGR2RGB)
    pred_overlay = cv2.cvtColor(overlay_heatmap(raw_bgr, pred_heat_np), cv2.COLOR_BGR2RGB)
    before_overlay = cv2.cvtColor(overlay_heatmap(raw_bgr, feat_before), cv2.COLOR_BGR2RGB)
    after_overlay = cv2.cvtColor(overlay_heatmap(raw_bgr, feat_after), cv2.COLOR_BGR2RGB)

    # 单独保存
    cv2.imwrite(os.path.join(args.output_dir, "raw.jpg"), cv2.cvtColor(raw_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(args.output_dir, "gt_heat_overlay.jpg"), cv2.cvtColor(gt_overlay, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(args.output_dir, "pred_heat_overlay.jpg"), cv2.cvtColor(pred_overlay, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(args.output_dir, "feat_before_overlay.jpg"), cv2.cvtColor(before_overlay, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(args.output_dir, "feat_after_overlay.jpg"), cv2.cvtColor(after_overlay, cv2.COLOR_RGB2BGR))

    # 拼总图
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    titles = [
        "Input Image",
        "GT Gaussian Heatmap",
        "Predicted Heatmap",
        "Before PGD",
        "After PGD",
    ]
    imgs = [raw_rgb, gt_overlay, pred_overlay, before_overlay, after_overlay]

    for ax, im, title in zip(axes, imgs, titles):
        ax.imshow(im)
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    save_path = os.path.join(args.output_dir, "pgd_grid.jpg")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"[Saved] {save_path}")


if __name__ == "__main__":
    main()
