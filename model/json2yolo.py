import os
import json
import glob
from tqdm import tqdm
from PIL import Image

# 类别映射（你可以根据自己的类别修改）
label_map = {
    "pack": 0,
    "piper": 1,
    "tip": 2,
    "tip_1": 3,
    "tip_2": 4,
}

# input_dir = "./images/train"      # JSON 文件夹路径
# output_dir = "./labels/train"       # 输出 YOLO 标签文件夹

input_dir = "./images/val"      # JSON 文件夹路径
output_dir = "./labels/val"       # 输出 YOLO 标签文件夹

os.makedirs(output_dir, exist_ok=True)

for json_path in tqdm(glob.glob(os.path.join(input_dir, "*.json"))):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 获取图像尺寸
    img_name = os.path.basename(data["imagePath"])
    image_path = os.path.join(input_dir, img_name)
    # image_path = os.path.join(os.path.dirname(json_path), data["imagePath"])
    image = Image.open(image_path)
    w, h = image.size

    label_lines = []
    for shape in data["shapes"]:
        label_name = shape["label"]
        if label_name not in label_map:
            continue
        cls_id = label_map[label_name]

        points = shape["points"]

        # 将多边形点归一化
        norm_points = []
        for (x, y) in points:
            norm_points.append(x / w)
            norm_points.append(y / h)

        coords_str = " ".join([f"{p:.6f}" for p in norm_points])
        label_lines.append(f"{cls_id} {coords_str}\n")

    # 写入 YOLO 格式标签
    base_name = os.path.splitext(os.path.basename(json_path))[0]
    with open(os.path.join(output_dir, f"{base_name}.txt"), "w") as f:
        f.writelines(label_lines)
