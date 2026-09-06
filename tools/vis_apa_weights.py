# tools/vis_apa_weights.py
import argparse
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from ultralytics import YOLO
from ultralytics.nn.modules.pa_module import PAFeature

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize APA spatial fusion weights.")
    parser.add_argument("--weights", default="runs/pgap_yolo_s_200e/weights/best.pt")
    parser.add_argument("--image", default="vis_inputs/dense_all_0000295_02400_d_0000033.jpg")
    parser.add_argument("--output-dir", default="vis_outputs/apa")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0", help="Ultralytics device string, for example 0 or cpu.")
    return parser.parse_args()


def normalize_map(x):
    x = x.astype(np.float32)
    x = x - x.min()
    if x.max() > 1e-6:
        x = x / x.max()
    return x


def heatmap_overlay(img_bgr, heat):
    heat = normalize_map(heat)
    heat_u8 = np.uint8(255 * heat)
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_bgr, 0.55, heat_color, 0.45, 0)
    return overlay


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    model = YOLO(args.weights)
    net = model.model  # DetectionModel

    # 打开 APA 权重缓存
    pa_modules = []
    for m in net.modules():
        if isinstance(m, PAFeature):
            m.save_weight = True
            pa_modules.append(m)

    # 推理一次
    _ = model.predict(
        source=args.image,
        imgsz=args.imgsz,
        conf=0.25,
        iou=0.7,
        device=args.device,
        save=False,
        verbose=False
    )

    # 读取原图
    img_bgr = cv2.imread(args.image)
    H, W = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 按 target 排序：0->PA-P2, 1->PA-P3, 2->PA-P4
    pa_modules = sorted(pa_modules, key=lambda x: x.target)

    # 画总图：4行3列
    fig, axes = plt.subplots(4, 3, figsize=(12, 14))

    source_names = ["from P2", "from P3", "from P4"]
    target_names = ["PA-P2", "PA-P3", "PA-P4"]

    # 第一行：原图重复一行，也可以只放第一张
    for j in range(3):
        axes[0, j].imshow(img_rgb)
        axes[0, j].set_title(f"Input Image ({target_names[j]})")
        axes[0, j].axis("off")

    # 后三行：每个目标尺度的3个来源权重
    for i, m in enumerate(pa_modules):  # i = 0,1,2
        w = m.last_weight  # [1, 3, h, w]
        if w is None:
            print(f"[Warning] No saved weight for target={m.target}")
            continue

        w = w[0].cpu().numpy()  # [3, h, w]

        for s in range(3):
            heat = w[s]
            heat = cv2.resize(heat, (W, H), interpolation=cv2.INTER_LINEAR)
            overlay = heatmap_overlay(img_bgr, heat)
            overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

            axes[i + 1, s].imshow(overlay)
            axes[i + 1, s].set_title(f"{target_names[i]} {source_names[s]}")
            axes[i + 1, s].axis("off")

            # 单独保存
            save_single = os.path.join(
                args.output_dir, f"{target_names[i]}_{source_names[s].replace(' ', '_')}.jpg"
            )
            cv2.imwrite(save_single, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    plt.tight_layout()
    save_path = os.path.join(args.output_dir, "apa_weight_grid.jpg")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"[Saved] {save_path}")


if __name__ == "__main__":
    main()
