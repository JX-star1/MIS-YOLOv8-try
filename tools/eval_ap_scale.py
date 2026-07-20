import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics import YOLO


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def load_data_yaml(data_yaml):
    from pathlib import Path
    import yaml

    data_yaml = Path(data_yaml)
    if data_yaml.is_dir():
        raise IsADirectoryError(f"--data should be a yaml file, not directory: {data_yaml}")

    with open(data_yaml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    root = Path(data.get("path", data_yaml.parent))

    if not root.is_absolute():
        candidates = [
            (Path.cwd() / root).resolve(),
            (data_yaml.parent / root).resolve(),
            (Path.cwd() / "datasets" / root.name).resolve(),
        ]

        if len(data_yaml.parents) >= 4:
            candidates.append((data_yaml.parents[3] / "datasets" / root.name).resolve())

        for c in candidates:
            if c.exists():
                root = c
                break
        else:
            root = (Path.cwd() / "datasets" / root.name).resolve()

    names = data.get("names", {})
    if isinstance(names, list):
        names = {i: n for i, n in enumerate(names)}

    return data, root, names


def resolve_split_path(data, base, split):
    split_value = data.get(split, None)
    if split_value is None:
        raise ValueError(f"Split '{split}' not found in data yaml.")

    split_path = Path(split_value)
    if not split_path.is_absolute():
        split_path = (base / split_path).resolve()

    return split_path


def collect_images(split_path):
    split_path = Path(split_path)

    if split_path.is_file() and split_path.suffix == ".txt":
        imgs = []
        with open(split_path, "r", encoding="utf-8") as f:
            for line in f:
                p = Path(line.strip())
                if p.exists():
                    imgs.append(p.resolve())
        return sorted(imgs)

    if split_path.is_dir():
        imgs = [p.resolve() for p in split_path.rglob("*") if p.suffix.lower() in IMG_EXTS]
        return sorted(imgs)

    raise FileNotFoundError(f"Cannot find split path: {split_path}")


def infer_label_path(img_path):
    """
    Infer YOLO label path from image path.
    Example:
      .../images/val/000001.jpg -> .../labels/val/000001.txt
    """
    img_path = Path(img_path)
    parts = list(img_path.parts)

    if "images" in parts:
        idx = parts.index("images")
        parts[idx] = "labels"
        return Path(*parts).with_suffix(".txt")

    # fallback
    return img_path.with_suffix(".txt")


def build_coco_gt(image_files, names):
    images = []
    annotations = []
    categories = [{"id": i + 1, "name": name} for i, name in enumerate(names)]

    image_id_map = {}
    ann_id = 1

    for img_id, img_path in enumerate(image_files, start=1):
        img_path = Path(img_path)
        with Image.open(img_path) as im:
            w_img, h_img = im.size

        images.append({
            "id": img_id,
            "file_name": str(img_path),
            "width": w_img,
            "height": h_img,
        })
        image_id_map[str(img_path.resolve())] = img_id

        label_path = infer_label_path(img_path)
        if not label_path.exists():
            continue

        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                items = line.strip().split()
                if len(items) < 5:
                    continue

                cls = int(float(items[0]))
                cx = float(items[1])
                cy = float(items[2])
                bw = float(items[3])
                bh = float(items[4])

                # YOLO normalized xywh -> pixel xywh
                x = (cx - bw / 2.0) * w_img
                y = (cy - bh / 2.0) * h_img
                box_w = bw * w_img
                box_h = bh * h_img

                # Clamp to image boundary
                x = max(0.0, min(x, w_img - 1.0))
                y = max(0.0, min(y, h_img - 1.0))
                box_w = max(1.0, min(box_w, w_img - x))
                box_h = max(1.0, min(box_h, h_img - y))

                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cls + 1,
                    "bbox": [x, y, box_w, box_h],
                    "area": box_w * box_h,
                    "iscrowd": 0,
                })
                ann_id += 1

    gt = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    return gt, image_id_map


def run_predictions(weights, image_files, image_id_map, imgsz, batch, device, conf, nms_iou, max_det):
    import gc
    import torch
    from pathlib import Path
    from ultralytics import YOLO

    model = YOLO(weights)
    detections = []

    device_str = str(device).lower()
    use_half = device_str != "cpu"

    for idx, img_path in enumerate(image_files, start=1):
        img_path = Path(img_path)
        img_key = str(img_path.resolve())
        img_id = image_id_map[img_key]

        results = model.predict(
            source=str(img_path),
            imgsz=imgsz,
            batch=1,
            device=device,
            conf=conf,
            iou=nms_iou,
            max_det=max_det,
            save=False,
            verbose=False,
            stream=False,
            half=use_half
        )

        r = results[0]

        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.detach().cpu().numpy()
            scores = r.boxes.conf.detach().cpu().numpy()
            clss = r.boxes.cls.detach().cpu().numpy().astype(int)

            for box, score, cls in zip(xyxy, scores, clss):
                x1, y1, x2, y2 = box.tolist()
                w = max(0.0, x2 - x1)
                h = max(0.0, y2 - y1)

                if w <= 0 or h <= 0:
                    continue

                detections.append({
                    "image_id": img_id,
                    "category_id": int(cls) + 1,
                    "bbox": [float(x1), float(y1), float(w), float(h)],
                    "score": float(score),
                })

        del results, r

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        gc.collect()

        if idx % 50 == 0 or idx == len(image_files):
            print(f"[INFO] Predicted {idx}/{len(image_files)} images")

    return detections


def compute_area_ap(coco_gt, detections, output_dir, small_thr, medium_thr, max_det):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gt_json = output_dir / "gt_coco.json"
    dt_json = output_dir / "pred_coco.json"

    with open(gt_json, "w", encoding="utf-8") as f:
        json.dump(coco_gt, f)

    with open(dt_json, "w", encoding="utf-8") as f:
        json.dump(detections, f)

    coco = COCO(str(gt_json))
    coco_dt = coco.loadRes(str(dt_json)) if len(detections) > 0 else coco.loadRes([])

    coco_eval = COCOeval(coco, coco_dt, iouType="bbox")
    coco_eval.params.maxDets = [1, 10, max_det]

    # COCO default: small < 32^2, medium 32^2~96^2, large > 96^2
    # You can change thresholds by args.
    coco_eval.params.areaRng = [
        [0 ** 2, 1e10],
        [0 ** 2, small_thr ** 2],
        [small_thr ** 2, medium_thr ** 2],
        [medium_thr ** 2, 1e10],
    ]
    coco_eval.params.areaRngLbl = ["all", "small", "medium", "large"]

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return coco_eval


def mean_ap(coco_eval, iou=None, area_label="all", max_det=300):
    """
    Get AP from COCOeval precision array.

    precision shape:
    [T, R, K, A, M]
    T: IoU thresholds
    R: recall thresholds
    K: categories
    A: area ranges
    M: max detections
    """
    p = coco_eval.params
    precision = coco_eval.eval["precision"]

    area_idx = list(p.areaRngLbl).index(area_label)
    max_det_idx = list(p.maxDets).index(max_det)

    if iou is None:
        pr = precision[:, :, :, area_idx, max_det_idx]
    else:
        iou_idx = np.where(np.isclose(p.iouThrs, iou))[0]
        if len(iou_idx) == 0:
            raise ValueError(f"IoU {iou} not found in COCOeval thresholds.")
        pr = precision[iou_idx, :, :, area_idx, max_det_idx]

    pr = pr[pr > -1]
    return float(np.mean(pr)) if pr.size else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--nms-iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--small-thr", type=int, default=32)
    parser.add_argument("--medium-thr", type=int, default=96)
    parser.add_argument("--out", type=str, default="runs_scale_ap")
    parser.add_argument("--method", type=str, default=None)
    args = parser.parse_args()

    data, base, names = load_data_yaml(args.data)
    split_path = resolve_split_path(data, base, args.split)
    image_files = collect_images(split_path)

    if len(image_files) == 0:
        raise RuntimeError(f"No images found in {split_path}")

    method = args.method or Path(args.weights).parents[1].name
    output_dir = Path(args.out) / method

    print(f"[INFO] Method: {method}")
    print(f"[INFO] Weights: {args.weights}")
    print(f"[INFO] Data: {args.data}")
    print(f"[INFO] Split: {args.split}")
    print(f"[INFO] Images: {len(image_files)}")
    print(f"[INFO] Area thresholds: small < {args.small_thr}^2, medium < {args.medium_thr}^2")

    print("[INFO] Building COCO-format ground truth...")
    coco_gt, image_id_map = build_coco_gt(image_files, names)

    print("[INFO] Running model predictions...")
    detections = run_predictions(
        weights=args.weights,
        image_files=image_files,
        image_id_map=image_id_map,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        conf=args.conf,
        nms_iou=args.nms_iou,
        max_det=args.max_det,
    )

    print(f"[INFO] Number of detections: {len(detections)}")
    print("[INFO] Computing AP_S / AP_M / AP_L...")
    coco_eval = compute_area_ap(
        coco_gt=coco_gt,
        detections=detections,
        output_dir=output_dir,
        small_thr=args.small_thr,
        medium_thr=args.medium_thr,
        max_det=args.max_det,
    )

    metrics = {
        "method": method,
        "AP50_all": mean_ap(coco_eval, iou=0.50, area_label="all", max_det=args.max_det) * 100,
        "AP50_S": mean_ap(coco_eval, iou=0.50, area_label="small", max_det=args.max_det) * 100,
        "AP50_M": mean_ap(coco_eval, iou=0.50, area_label="medium", max_det=args.max_det) * 100,
        "AP50_L": mean_ap(coco_eval, iou=0.50, area_label="large", max_det=args.max_det) * 100,
        "AP_50_95_all": mean_ap(coco_eval, iou=None, area_label="all", max_det=args.max_det) * 100,
        "AP_S": mean_ap(coco_eval, iou=None, area_label="small", max_det=args.max_det) * 100,
        "AP_M": mean_ap(coco_eval, iou=None, area_label="medium", max_det=args.max_det) * 100,
        "AP_L": mean_ap(coco_eval, iou=None, area_label="large", max_det=args.max_det) * 100,
    }

    print("\n================ Scale-specific AP Results ================")
    for k, v in metrics.items():
        if k == "method":
            print(f"{k}: {v}")
        else:
            print(f"{k}: {v:.3f}")

    csv_path = Path(args.out) / "scale_ap_results.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    header = list(metrics.keys())
    write_header = not csv_path.exists()

    with open(csv_path, "a", encoding="utf-8") as f:
        if write_header:
            f.write(",".join(header) + "\n")
        f.write(",".join(str(metrics[h]) for h in header) + "\n")

    print(f"\n[INFO] Results saved to: {csv_path}")



if __name__ == "__main__":
    main()
