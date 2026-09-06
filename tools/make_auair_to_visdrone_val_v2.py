import json
import shutil
from pathlib import Path
from collections import Counter

CLASS_MAP = {
    0:0,    # Human -> pedestrian
    1:3,    # Car
    2:5,    # Truck
    3:4,    # Van
    4:9,    # Motorbike
    5:2,    # Bicycle
    6:8,    # Bus
    7:-1    # Trailer ignore
}

json_file="au_air/raw/annotations.json"
img_root=Path("au_air/raw/images")

save_root=Path("au_air/to_visdrone")
img_save=save_root/"images"/"val"
lab_save=save_root/"labels"/"val"

img_save.mkdir(parents=True,exist_ok=True)
lab_save.mkdir(parents=True,exist_ok=True)

with open(json_file,"r") as f:
    data=json.load(f)

anns=data["annotations"]

cnt=Counter()
mapped=0
linked=0

for ann in anns:

    img_name=ann["image_name"]
    src=img_root/img_name

    if not src.exists():
        continue

    linked+=1

    shutil.copy2(src,img_save/img_name)

    W=float(ann["image_width"])
    H=float(ann["image_height"])

    lines=[]

    for obj in ann["bbox"]:

        cls=CLASS_MAP.get(obj["class"],-1)

        if cls<0:
            continue

        left=float(obj["left"])
        top=float(obj["top"])
        width=float(obj["width"])
        height=float(obj["height"])

        xc=(left+width/2)/W
        yc=(top+height/2)/H
        bw=width/W
        bh=height/H

        lines.append(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

        mapped+=1
        cnt[cls]+=1

    with open(lab_save/(Path(img_name).stem+".txt"),"w") as f:
        f.write("\n".join(lines))

yaml=save_root/"AU-AIR_to_VisDrone.yaml"

yaml.write_text("""path: au_air/to_visdrone

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

print("="*60)
print("images linked :",linked)
print("labels written:",linked)
print("mapped boxes  :",mapped)
print("="*60)

names=[
"pedestrian",
"people",
"bicycle",
"car",
"van",
"truck",
"tricycle",
"awning-tricycle",
"bus",
"motor"
]

for k in sorted(cnt):
    print(names[k],cnt[k])
