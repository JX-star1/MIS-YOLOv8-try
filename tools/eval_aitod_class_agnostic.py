import argparse
from pathlib import Path
import gc
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
import torch


def xywhn_to_xyxy(box, w, h):
    x, y, bw, bh = box
    return [
        (x - bw / 2) * w,
        (y - bh / 2) * h,
        (x + bw / 2) * w,
        (y + bh / 2) * h,
    ]


def box_iou_one_to_many(box, boxes):
    if len(boxes) == 0:
        return np.zeros((0,), dtype=np.float32)

    box = np.array(box, dtype=np.float32)
    boxes = np.array(boxes, dtype=np.float32)

    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    area1 = max(0, box[2] - box[0]) * max(0, box[3] - box[1])
    area2 = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])

    union = area1 + area2 - inter + 1e-9
    return inter / union


def compute_ap(rec, prec):
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))

    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    x = np.linspace(0, 1, 101)
    return np.trapz(np.interp(x, mrec, mpre), x)


def eval_at_iou(preds, gts, iou_thr):
    preds = sorted(preds, key=lambda x: x[1], reverse=True)
    n_gt = sum(len(v) for v in gts.values())

    matched = {k: np.zeros(len(v), dtype=bool) for k, v in gts.items()}

    tp = np.zeros(len(preds), dtype=np.float32)
    fp = np.zeros(len(preds), dtype=np.float32)

    for i, (img_name, score, box) in enumerate(preds):
        gt_boxes = gts.get(img_name, [])

        if len(gt_boxes) == 0:
            fp[i] = 1
            continue

        ious = box_iou_one_to_many(box, gt_boxes)

        for j in range(len(ious)):
            if matched[img_name][j]:
                ious[j] = -1

        best_idx = int(np.argmax(ious))
        best_iou = float(ious[best_idx])

        if best_iou >= iou_thr:
            tp[i] = 1
            matched[img_name][best_idx] = True
        else:
            fp[i] = 1

    if n_gt == 0 or len(preds) == 0:
        return 0.0, 0.0, 0.0

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)

    rec = tp_cum / (n_gt + 1e-9)
    prec = tp_cum / (tp_cum + fp_cum + 1e-9)

    ap = compute_ap(rec, prec)
    final_r = rec[-1] if len(rec) else 0.0
    final_p = prec[-1] if len(prec) else 0.0

    return float(ap), float(final_p), float(final_r)


def load_gt_for_image(label_path, w, h):
    boxes = []
    if not label_path.exists():
        return boxes

    for line in label_path.read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) < 5:
            continue

        x, y, bw, bh = map(float, parts[1:5])
        boxes.append(xywhn_to_xyxy([x, y, bw, bh], w, h))

    return boxes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--images", default="aitod/yolo/images/val")
    parser.add_argument("--labels", default="aitod/yolo/labels/val")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=100)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    images_dir = Path(args.images)
    labels_dir = Path(args.labels)

    image_files = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"]:
        image_files.extend(images_dir.glob(ext))
    image_files = sorted(image_files)

    if not image_files:
        raise FileNotFoundError(f"No images found in {images_dir}")

    model = YOLO(args.weights)

    preds = []
    gts = {}

    for img_path in tqdm(image_files, desc="Predict one by one"):
        # 一张图一张图预测，避免 Ultralytics 内部一次性占大显存
        results = model.predict(
            source=str(img_path),
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            device=args.device,
            batch=1,
            verbose=False,
            agnostic_nms=True,
            stream=False,
        )

        r = results[0]
        img_name = Path(r.path).name
        h, w = r.orig_shape

        label_path = labels_dir / f"{Path(r.path).stem}.txt"
        gts[img_name] = load_gt_for_image(label_path, w, h)

        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.detach().cpu().numpy()
            conf = r.boxes.conf.detach().cpu().numpy()

            for b, s in zip(xyxy, conf):
                preds.append((img_name, float(s), b.tolist()))

        del results
        del r
        gc.collect()
        if torch.cuda.is_available() and args.device != "cpu":
            torch.cuda.empty_cache()

    n_gt = sum(len(v) for v in gts.values())

    print("=" * 80)
    print("weights:", args.weights)
    print("images:", len(image_files))
    print("gt boxes:", n_gt)
    print("pred boxes:", len(preds))

    thresholds = np.arange(0.5, 0.96, 0.05)
    aps = []

    for t in thresholds:
        ap, p, r = eval_at_iou(preds, gts, float(t))
        aps.append(ap)
        print(f"AP@{t:.2f}: {ap:.4f} | P: {p:.4f} | R: {r:.4f}")

    map50 = aps[0]
    map5095 = float(np.mean(aps))

    print("=" * 80)
    print(f"Class-agnostic mAP@0.5:      {map50:.4f}")
    print(f"Class-agnostic mAP@0.5:0.95: {map5095:.4f}")


if __name__ == "__main__":
    main()
