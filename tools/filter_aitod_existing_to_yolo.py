import argparse
import json
from pathlib import Path
from tqdm import tqdm


def convert(split, ann_file, img_dir, output_root):
    ann_file = Path(ann_file)
    img_dir = Path(img_dir)

    data = json.load(open(ann_file, "r"))

    cats = sorted(data["categories"], key=lambda x: x["id"])
    cat_id_to_new = {c["id"]: i for i, c in enumerate(cats)}
    names = [c["name"] for c in cats]

    existing = {}
    for p in img_dir.rglob("*"):
        if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]:
            existing[p.name] = p.resolve()

    anns_by_img = {}
    for ann in data["annotations"]:
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    img_out = output_root / "images" / split
    lab_out = output_root / "labels" / split
    img_out.mkdir(parents=True, exist_ok=True)
    lab_out.mkdir(parents=True, exist_ok=True)

    used_imgs = 0
    used_boxes = 0
    missing = 0

    for im in tqdm(data["images"], desc=split):
        name = Path(im["file_name"]).name

        if name not in existing:
            missing += 1
            continue

        src = existing[name]
        dst = img_out / name

        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src)

        w_img = im["width"]
        h_img = im["height"]

        lines = []
        for ann in anns_by_img.get(im["id"], []):
            if ann.get("iscrowd", 0) == 1:
                continue

            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue

            cls = cat_id_to_new[ann["category_id"]]

            xc = (x + w / 2.0) / w_img
            yc = (y + h / 2.0) / h_img
            bw = w / w_img
            bh = h / h_img

            xc = min(max(xc, 0.0), 1.0)
            yc = min(max(yc, 0.0), 1.0)
            bw = min(max(bw, 0.0), 1.0)
            bh = min(max(bh, 0.0), 1.0)

            if bw <= 0 or bh <= 0:
                continue

            lines.append(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
            used_boxes += 1

        with open(lab_out / f"{Path(name).stem}.txt", "w") as f:
            f.writelines(lines)

        used_imgs += 1

    with open(output_root / "names.txt", "w") as f:
        for n in names:
            f.write(n + "\n")

    print("=" * 60)
    print(f"{split}")
    print(f"json images: {len(data['images'])}")
    print(f"existing images: {len(existing)}")
    print(f"used images: {used_imgs}")
    print(f"missing images: {missing}")
    print(f"boxes: {used_boxes}")
    print(f"classes: {names}")


def main():
    parser = argparse.ArgumentParser(description="Convert existing AI-TOD COCO images and labels to YOLO format.")
    parser.add_argument("--source-root", type=Path, default=Path("data/aitod/raw"))
    parser.add_argument("--output-root", type=Path, default=Path("data/aitod/yolo"))
    args = parser.parse_args()

    convert(
        "train",
        args.source_root / "aitod/annotations/aitod_train.json",
        args.source_root / "aitod/images/train",
        args.output_root,
    )

    convert(
        "val",
        args.source_root / "aitod/annotations/aitod_val.json",
        args.source_root / "aitod/images/val",
        args.output_root,
    )

    names = [x.strip() for x in open(args.output_root / "names.txt") if x.strip()]

    yaml_path = args.output_root / "AI-TOD.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"path: {args.output_root.resolve()}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("names:\n")
        for i, n in enumerate(names):
            f.write(f"  {i}: {n}\n")

    print("=" * 60)
    print(open(yaml_path).read())


if __name__ == "__main__":
    main()
