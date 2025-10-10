import cv2
import numpy as np
from ultralytics import YOLO

# === 加载模型 ===
model = YOLO("./runs/segment/exp_yolo11_seg/weights/best.pt")  # 替换为你的路径

# === 读取测试图像 ===
img_path = "./unlabled/zed_left_1759898032877.jpg"  # 你要测试的图片
img = cv2.imread(img_path)

# === 执行推理 ===
results = model(img, verbose=False)

yolo_frame = results[0].plot()

cv2.imshow("YOLO Frame", yolo_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
