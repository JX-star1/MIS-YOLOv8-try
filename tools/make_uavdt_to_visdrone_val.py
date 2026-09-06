import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Map a UAVDT validation set to VisDrone classes.")
    parser.add_argument("--source-images", type=Path, default=Path("data/uavdt/raw/UAVDT/UAVDT/val/images"))
    parser.add_argument("--source-labels", type=Path, default=Path("data/uavdt/raw/UAVDT/UAVDT/val/labels"))
    parser.add_argument("--output-root", type=Path, default=Path("data/uavdt/to_visdrone"))
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

    cls_map = {
        0: 3,  # car
        1: 5,  # truck
        2: 8,  # bus
    }

    img_exts = {".jpg", ".jpeg", ".png", ".bmp"}

    n_img = 0
    n_lab = 0
    n_box = 0

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
                parts[0] = str(cls_map[old_cls])
                lines_out.append(" ".join(parts) + "\n")
                n_box += 1

        dst_txt.write_text("".join(lines_out))
        n_lab += 1

    yaml_path = out / "UAVDT_to_VisDrone.yaml"
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
    print("yaml:", yaml_path)


if __name__ == "__main__":
    main()
