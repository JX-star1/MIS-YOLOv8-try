import argparse
import json
from pathlib import Path
from collections import Counter
from PIL import Image


# AU-AIR -> VisDrone 类别映射
NAME_MAP = {
    "human": 0,       # pedestrian
    "person": 0,
    "pedestrian": 0,

    "bike": 2,        # bicycle
    "bicycle": 2,

    "car": 3,
    "van": 4,
    "truck": 5,
    "bus": 8,

    "motorbike": 9,   # motor
    "motorcycle": 9,
    "motor": 9,

    # trailer / trailar 没有严格对应，忽略
}


def find_images(img_root):
    img_root = Path(img_root)
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return {p.name: p.resolve() for p in img_root.rglob("*") if p.suffix.lower() in exts}


def get_img_size(path):
    with Image.open(path) as im:
        return im.size  # w, h


def normalize_bbox_from_dict(obj, w_img, h_img):
    # 支持 xmin/ymin/xmax/ymax
    if all(k in obj for k in ["xmin", "ymin", "xmax", "ymax"]):
        x1 = float(obj["xmin"])
        y1 = float(obj["ymin"])
        x2 = float(obj["xmax"])
        y2 = float(obj["ymax"])
        bw = x2 - x1
        bh = y2 - y1

    # 支持 x/y/w/h 或 left/top/width/height
    elif all(k in obj for k in ["x", "y", "w", "h"]):
        x1 = float(obj["x"])
        y1 = float(obj["y"])
        bw = float(obj["w"])
        bh = float(obj["h"])

    elif all(k in obj for k in ["left", "top", "width", "height"]):
        x1 = float(obj["left"])
        y1 = float(obj["top"])
        bw = float(obj["width"])
        bh = float(obj["height"])

    elif "bbox" in obj:
        b = obj["bbox"]
        if isinstance(b, dict):
            return normalize_bbox_from_dict(b, w_img, h_img)
        if not isinstance(b, (list, tuple)) or len(b) != 4:
            return None

        # 默认按 [x, y, w, h] 处理。AU-AIR 常见转换就是这种逻辑。
        x1, y1, bw, bh = map(float, b)

    else:
        return None

    if bw <= 0 or bh <= 0:
        return None

    xc = (x1 + bw / 2.0) / w_img
    yc = (y1 + bh / 2.0) / h_img
    bw = bw / w_img
    bh = bh / h_img

    xc = min(max(xc, 0.0), 1.0)
    yc = min(max(yc, 0.0), 1.0)
    bw = min(max(bw, 0.0), 1.0)
    bh = min(max(bh, 0.0), 1.0)

    if bw <= 0 or bh <= 0:
        return None

    return xc, yc, bw, bh


def get_label(obj):
    for k in ["class", "category", "label", "name", "class_name", "category_name"]:
        if k in obj:
            return str(obj[k]).lower().strip()
    return None


def get_frame_name(frame):
    for k in ["image_name", "file_name", "filename", "image", "img_name", "name"]:
        if k in frame:
            return Path(str(frame[k])).name
    return None


def parse_frame_level(data):
    if isinstance(data, dict):
        if "annotations" in data and isinstance(data["annotations"], list):
            return data["annotations"]
        if "images" in data and isinstance(data["images"], list):
            return data["images"]
    elif isinstance(data, list):
        return data
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann", required=True, help="AU-AIR annotation json")
    parser.add_argument("--images", required=True, help="AU-AIR image root")
    parser.add_argument("--out", default="au_air/to_visdrone")
    args = parser.parse_args()

    ann_path = Path(args.ann)
    img_root = Path(args.images)
    out = Path(args.out)

    out_img = out / "images/val"
    out_lab = out / "labels/val"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lab.mkdir(parents=True, exist_ok=True)

    image_index = find_images(img_root)
    print("indexed images:", len(image_index))

    data = json.load(open(ann_path, "r"))
    frames = parse_frame_level(data)

    if frames is None:
        raise RuntimeError("Unsupported annotation format. Please show the first 30 lines of the json.")

    n_img = 0
    n_lab = 0
    n_box = 0
    missing_img = 0
    cnt = Counter()

    for frame in frames:
        img_name = get_frame_name(frame)
        if img_name is None:
            continue

        if img_name not in image_index:
            missing_img += 1
            continue

        src_img = image_index[img_name]
        dst_img = out_img / img_name

        if dst_img.exists() or dst_img.is_symlink():
            dst_img.unlink()
        dst_img.symlink_to(src_img)

        # 获取图像尺寸
        w_img = frame.get("width") or frame.get("image_width")
        h_img = frame.get("height") or frame.get("image_height")
        if w_img is None or h_img is None:
            w_img, h_img = get_img_size(src_img)
        else:
            w_img, h_img = int(w_img), int(h_img)

        # AU-AIR 通常是每张图里有 annotations/objects 列表
        objs = frame.get("annotations") or frame.get("objects") or frame.get("bboxes") or []
        if isinstance(objs, dict):
            objs = [objs]

        lines = []
        for obj in objs:
            label = get_label(obj)
            if label is None:
                continue

            label = label.lower().strip()
            if label not in NAME_MAP:
                continue

            cls = NAME_MAP[label]
            box = normalize_bbox_from_dict(obj, w_img, h_img)
            if box is None:
                continue

            xc, yc, bw, bh = box
            lines.append(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
            cnt[cls] += 1
            n_box += 1

        (out_lab / f"{Path(img_name).stem}.txt").write_text("".join(lines))
        n_img += 1
        n_lab += 1

    yaml_path = out / "AU-AIR_to_VisDrone.yaml"
    yaml_path.write_text(f"""path: {out.resolve()}
train: images/val
val: images/val

names:
  0: pedestrian
  1: people
  2: bicycle
  3: car
  4: van
  5: truck
  6: tricycle
  7: awning-tricycle
  8: bus
  9: motor
""")

    print("=" * 80)
    print("images linked:", n_img)
    print("labels written:", n_lab)
    print("mapped boxes:", n_box)
    print("missing images:", missing_img)
    print("class counts:", cnt)
    print("yaml:", yaml_path)


if __name__ == "__main__":
    main()
