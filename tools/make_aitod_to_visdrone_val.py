import argparse
from pathlib import Path
from collections import Counter


def parse_args():
    parser = argparse.ArgumentParser(description="Map an AI-TOD validation set to VisDrone classes.")
    parser.add_argument("--source-images", type=Path, default=Path("data/aitod/yolo/images/val"))
    parser.add_argument("--source-labels", type=Path, default=Path("data/aitod/yolo/labels/val"))
    parser.add_argument("--output-root", type=Path, default=Path("data/aitod/to_visdrone"))
    return parser.parse_args()


def main():
    args = parse_args()
    src_img = args.source_images
    src_lab = args.source_labels
    out = args.output_root
    out_img = out / "images/val"
    out_lab = out / "labels/val"

    out_img.mkdir(parents=True, exist_ok=True)
    out_lab.mkdir(parents=True, exist_ok=True)

    if not src_img.exists():
        raise FileNotFoundError(f"Cannot find image dir: {src_img}")
    if not src_lab.exists():
        raise FileNotFoundError(f"Cannot find label dir: {src_lab}")

    # AI-TOD -> VisDrone
    cls_map = {
        5: 3,  # AI-TOD vehicle -> VisDrone car
        6: 0,  # AI-TOD person  -> VisDrone pedestrian
    }

    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    n_img = 0
    n_lab = 0
    n_box = 0
    cnt = Counter()

    for img in src_img.iterdir():
        if img.suffix.lower() not in img_exts:
            continue

        dst_img = out_img / img.name
        if dst_img.exists() or dst_img.is_symlink():
            dst_img.unlink()
        dst_img.symlink_to(img.resolve())
        n_img += 1

        src_txt = src_lab / f"{img.stem}.txt"
        dst_txt = out_lab / f"{img.stem}.txt"

        lines_out = []
        if src_txt.exists():
            for line in src_txt.read_text().splitlines():
                if not line.strip():
                    continue

                parts = line.split()
                old_cls = int(float(parts[0]))

                if old_cls not in cls_map:
                    continue

                new_cls = cls_map[old_cls]
                parts[0] = str(new_cls)
                lines_out.append(" ".join(parts) + "\n")
                cnt[new_cls] += 1
                n_box += 1

        dst_txt.write_text("".join(lines_out))
        n_lab += 1

    yaml_path = out / "AI-TOD_to_VisDrone.yaml"
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

    print("images:", n_img)
    print("labels:", n_lab)
    print("boxes:", n_box)
    print("class count after mapping:", cnt)
    print("yaml:", yaml_path)


if __name__ == "__main__":
    main()
