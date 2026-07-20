import os
import cv2
from ultralytics import YOLO

# =========================
# 1. 改成你自己的权重路径
# =========================
MODEL_DICT = {
    "yolov8s": "runs/detect/yolov8s/weights/best.pt",
    "wo_apa": "runs/detect/pgap_wo_apa/weights/best.pt",
    "asff_like": "runs/detect/pgap_asff/weights/best.pt",
    "pgap_yolo": "runs/detect/pgap_apa/weights/best.pt",
}

IMG_DIR = "vis_inputs"
OUT_DIR = "vis_outputs/det_compare"

IMGSZ = 640
CONF = 0.25
IOU = 0.7
DEVICE = 0

os.makedirs(OUT_DIR, exist_ok=True)


def draw_and_save(model_name, weight_path, img_path):
    model = YOLO(weight_path)

    results = model.predict(
        source=img_path,
        imgsz=IMGSZ,
        conf=CONF,
        iou=IOU,
        device=DEVICE,
        save=False,
        verbose=False
    )

    r = results[0]
    im = r.plot(conf=True, labels=True, boxes=True)

    base = os.path.splitext(os.path.basename(img_path))[0]
    save_path = os.path.join(OUT_DIR, f"{base}_{model_name}.jpg")
    cv2.imwrite(save_path, im)
    print(f"[Saved] {save_path}")


def main():
    img_list = [
        os.path.join(IMG_DIR, x)
        for x in sorted(os.listdir(IMG_DIR))
        if x.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ]

    for img_path in img_list:
        for model_name, weight_path in MODEL_DICT.items():
            draw_and_save(model_name, weight_path, img_path)


if __name__ == "__main__":
    main()