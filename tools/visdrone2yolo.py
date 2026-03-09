
from PIL import Image

from pathlib import Path



def convert_one_split(split_dir: str):

    split_dir = Path(split_dir)

    img_dir = split_dir / "images"

    ann_dir = split_dir / "annotations"

    out_dir = split_dir / "labels_yolo"

    out_dir.mkdir(exist_ok=True)



    img_suffix = [".jpg", ".png", ".jpeg", ".JPG", ".PNG", ".JPEG"]

    ann_files = sorted(ann_dir.glob("*.txt"))

    print(f"[{split_dir.name}] annotations: {len(ann_files)}")



    missing_imgs = 0

    bad_lines = 0



    for ann_path in ann_files:

        stem = ann_path.stem



        # 找到对应图片

        img_path = None

        for suf in img_suffix:

            p = img_dir / f"{stem}{suf}"

            if p.exists():

                img_path = p

                break

        if img_path is None:

            missing_imgs += 1

            continue



        w_img, h_img = Image.open(img_path).size



        yolo_lines = []

        with open(ann_path, "r", encoding="utf-8") as f:

            for line in f:

                line = line.strip()

                if not line:

                    continue

                parts = line.split(",")

                if len(parts) < 6:

                    bad_lines += 1

                    continue



                x, y, w, h = map(float, parts[:4])

                cls = int(parts[5])  # VisDrone: 1~10 有效



                # 过滤无效框/忽略类

                if w <= 0 or h <= 0:

                    continue

                if cls < 1 or cls > 10:

                    continue



                # YOLO：class 从 0 开始，坐标归一化到 [0,1]

                cls_yolo = cls - 1

                cx = (x + w / 2.0) / w_img

                cy = (y + h / 2.0) / h_img

                ww = w / w_img

                hh = h / h_img



                # clamp

                cx = min(max(cx, 0.0), 1.0)

                cy = min(max(cy, 0.0), 1.0)

                ww = min(max(ww, 0.0), 1.0)

                hh = min(max(hh, 0.0), 1.0)



                yolo_lines.append(f"{cls_yolo} {cx:.6f} {cy:.6f} {ww:.6f} {hh:.6f}")



        out_path = out_dir / f"{stem}.txt"

        out_path.write_text("\n".join(yolo_lines), encoding="utf-8")



    print(f"[{split_dir.name}] done. missing_imgs={missing_imgs}, bad_lines={bad_lines}, out={out_dir}")



if __name__ == "__main__":

    convert_one_split("/root/datasets/VisDrone/VisDrone2019-DET-train")

