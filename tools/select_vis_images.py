import argparse
import csv
import shutil
from pathlib import Path

import cv2
import yaml


VISDRONE_NAMES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor"
]


def resolve_path(p, base):
    p = Path(p)
    if p.is_absolute():
        return p
    return (base / p).resolve()


def collect_images(source, base):
    source = resolve_path(source, base)

    if source.is_file() and source.suffix.lower() == ".txt":
        imgs = []
        for line in source.read_text().splitlines():
            line = line.strip()
            if line:
                imgs.append(resolve_path(line, base))
        return imgs

    if source.is_dir():
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        return [p for p in sorted(source.rglob("*")) if p.suffix.lower() in exts]

    raise FileNotFoundError(f"Cannot find image source: {source}")


def img_to_label_path(img_path):
    p = Path(img_path)
    parts = list(p.parts)

    if "images" in parts:
        idx = parts.index("images")
        parts[idx] = "labels"
        return Path(*parts).with_suffix(".txt")

    return p.parent.parent / "labels" / p.parent.name / (p.stem + ".txt")


def read_labels(label_path):
    labels = []
    if not label_path.exists():
        return labels

    for line in label_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        vals = line.split()
        if len(vals) < 5:
            continue
        cls = int(float(vals[0]))
        x, y, w, h = map(float, vals[1:5])
        labels.append((cls, x, y, w, h))
    return labels


def analyze_one(img_path, imgsz=640):
    label_path = img_to_label_path(img_path)
    labels = read_labels(label_path)

    counts = {i: 0 for i in range(10)}
    tiny = 0
    very_tiny = 0

    for cls, x, y, w, h in labels:
        if cls in counts:
            counts[cls] += 1

        # 按网络输入 640 后的尺度估计目标大小
        box_w = w * imgsz
        box_h = h * imgsz
        max_side = max(box_w, box_h)

        if max_side <= 32:
            tiny += 1
        if max_side <= 16:
            very_tiny += 1

    person_count = counts[0] + counts[1]
    vehicle_count = counts[3] + counts[4] + counts[5] + counts[8] + counts[9]
    confusing_count = counts[6] + counts[7]
    category_num = sum(1 for v in counts.values() if v > 0)
    total = len(labels)

    return {
        "img": str(img_path),
        "label": str(label_path),
        "total": total,
        "tiny": tiny,
        "very_tiny": very_tiny,
        "person": person_count,
        "vehicle": vehicle_count,
        "confusing": confusing_count,
        "category_num": category_num,
        "counts": counts,
    }


def safe_copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def make_contact_sheet(selected, out_path, thumb_w=320):
    thumbs = []
    for item in selected:
        img = cv2.imread(item["copied_img"])
        if img is None:
            continue

        h, w = img.shape[:2]
        scale = thumb_w / max(w, 1)
        thumb_h = int(h * scale)
        img = cv2.resize(img, (thumb_w, thumb_h))

        text1 = item["tag"]
        text2 = f"all={item['total']} tiny={item['tiny']} person={item['person']} vehicle={item['vehicle']} cls={item['category_num']}"

        pad = 50
        canvas = cv2.copyMakeBorder(img, pad, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        cv2.putText(canvas, text1, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(canvas, text2, (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        thumbs.append(canvas)

    if not thumbs:
        return

    cols = 2
    rows = (len(thumbs) + cols - 1) // cols
    max_h = max(t.shape[0] for t in thumbs)
    max_w = max(t.shape[1] for t in thumbs)

    sheet = 255 * __import__("numpy").ones((rows * max_h, cols * max_w, 3), dtype="uint8")

    for i, t in enumerate(thumbs):
        r = i // cols
        c = i % cols
        sheet[r * max_h:r * max_h + t.shape[0], c * max_w:c * max_w + t.shape[1]] = t

    cv2.imwrite(str(out_path), sheet)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="ultralytics/cfg/datasets/VisDrone.yaml")
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--out", default="vis_inputs")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--per", type=int, default=2)
    parser.add_argument("--max", type=int, default=8)
    args = parser.parse_args()

    data_yaml = Path(args.data).resolve()
    with open(data_yaml, "r") as f:
        data = yaml.safe_load(f)

    base = data_yaml.parent
    if data.get("path", None):
        base = resolve_path(data["path"], base)

    imgs = collect_images(data[args.split], base)
    print(f"[Info] Found {len(imgs)} images from {args.split}")

    records = []
    for img_path in imgs:
        try:
            records.append(analyze_one(img_path, imgsz=args.imgsz))
        except Exception as e:
            print(f"[Warning] skip {img_path}: {e}")

    # 不同类别的挑选规则
    groups = {
        "dense_all": sorted(records, key=lambda x: x["total"], reverse=True),
        "dense_person": sorted(records, key=lambda x: x["person"], reverse=True),
        "dense_vehicle": sorted(records, key=lambda x: x["vehicle"], reverse=True),
        "tiny_many": sorted(records, key=lambda x: x["tiny"], reverse=True),
        "very_tiny_many": sorted(records, key=lambda x: x["very_tiny"], reverse=True),
        "mixed_categories": sorted(records, key=lambda x: (x["category_num"], x["total"]), reverse=True),
        "tricycle_confusing": sorted(records, key=lambda x: x["confusing"], reverse=True),
    }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    selected = []
    used = set()

    for tag, items in groups.items():
        picked = 0
        for item in items:
            if item["img"] in used:
                continue
            if item["total"] <= 0:
                continue

            used.add(item["img"])
            picked += 1

            img_src = Path(item["img"])
            lab_src = Path(item["label"])

            new_stem = f"{tag}_{img_src.stem}"
            img_dst = out_dir / f"{new_stem}{img_src.suffix}"
            lab_dst = out_dir / f"{new_stem}.txt"

            safe_copy(img_src, img_dst)
            if lab_src.exists():
                safe_copy(lab_src, lab_dst)

            item = dict(item)
            item["tag"] = tag
            item["copied_img"] = str(img_dst)
            item["copied_label"] = str(lab_dst)
            selected.append(item)

            print(f"[Copy] {tag}: {img_dst} | all={item['total']} tiny={item['tiny']} person={item['person']} vehicle={item['vehicle']} cls={item['category_num']}")

            if picked >= args.per:
                break

            if len(selected) >= args.max:
                break

        if len(selected) >= args.max:
            break

    # 保存统计表
    csv_path = out_dir / "selected_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "tag", "copied_img", "copied_label", "total", "tiny",
                "very_tiny", "person", "vehicle", "confusing", "category_num"
            ]
        )
        writer.writeheader()
        for item in selected:
            writer.writerow({
                "tag": item["tag"],
                "copied_img": item["copied_img"],
                "copied_label": item["copied_label"],
                "total": item["total"],
                "tiny": item["tiny"],
                "very_tiny": item["very_tiny"],
                "person": item["person"],
                "vehicle": item["vehicle"],
                "confusing": item["confusing"],
                "category_num": item["category_num"],
            })

    sheet_path = Path("vis_outputs/contact_sheets/selected_vis_inputs.jpg")
    make_contact_sheet(selected, sheet_path)

    print(f"\n[Done] selected images saved to: {out_dir}")
    print(f"[Done] summary saved to: {csv_path}")
    print(f"[Done] contact sheet saved to: {sheet_path}")


if __name__ == "__main__":
    main()
