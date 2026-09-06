import argparse
import os

import cv2
from ultralytics import YOLO

DEFAULT_MODEL_DICT = {
    "wo_apa": "runs/pgap_yolov8s_noAPA_200e/weights/best.pt",
    "concat_fusion": "runs/pgap_yolov8s_concat_200e/weights/best.pt",
    "add_fusion": "runs/pgap_yolov8s_addfusion_200e/weights/best.pt",
    "bifpn_like": "runs/pgap_yolov8s_BiFPN-like_200e/weights/best.pt",
    "asff_like": "runs/pgap_yolov8s_asff-like_200e/weights/best.pt",
    "APA": "runs/pgap_yolo_s_200e/weights/best.pt",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Compare detections from PGAP-YOLO ablation checkpoints.")
    parser.add_argument("--input-dir", default="vis_inputs", help="Directory containing input images.")
    parser.add_argument("--output-dir", default="vis_outputs/det_compare", help="Directory for rendered images.")
    parser.add_argument(
        "--model-weight",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Override a checkpoint path; may be specified more than once.",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--device", default="0", help="Ultralytics device string, for example 0 or cpu.")
    return parser.parse_args()


def resolve_model_dict(overrides):
    model_dict = DEFAULT_MODEL_DICT.copy()
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid --model-weight value {item!r}; expected NAME=PATH")
        name, path = item.split("=", 1)
        if not name or not path:
            raise ValueError(f"Invalid --model-weight value {item!r}; expected NAME=PATH")
        model_dict[name] = path
    return model_dict


def draw_and_save(model_name, weight_path, img_path, output_dir, args):
    model = YOLO(weight_path)

    results = model.predict(
        source=img_path,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        save=False,
        verbose=False
    )

    r = results[0]
    im = r.plot(conf=True, labels=True, boxes=True)

    base = os.path.splitext(os.path.basename(img_path))[0]
    save_path = os.path.join(output_dir, f"{base}_{model_name}.jpg")
    cv2.imwrite(save_path, im)
    print(f"[Saved] {save_path}")


def main():
    args = parse_args()
    model_dict = resolve_model_dict(args.model_weight)
    os.makedirs(args.output_dir, exist_ok=True)

    img_list = [
        os.path.join(args.input_dir, x)
        for x in sorted(os.listdir(args.input_dir))
        if x.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ]

    for img_path in img_list:
        for model_name, weight_path in model_dict.items():
            draw_and_save(model_name, weight_path, img_path, args.output_dir, args)


if __name__ == "__main__":
    main()
