import cv2
import numpy as np

# === 参数设置 ===
img_path = "./left/zed_left_1760576521300.jpg"   # 你的图像路径
pattern_size = (9, 6)      # 棋盘内角点数 (列, 行)，注意是内角点数量！

# === 读取图像 ===
img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(f"找不到图像: {img_path}")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# === 检测棋盘角点 ===
ret, corners = cv2.findChessboardCorners(
    gray, pattern_size,
    cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
)

# === 如果检测到角点，进行亚像素级优化 ===
if ret:
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    cv2.drawChessboardCorners(img, pattern_size, corners_refined, ret)
    print(f"✅ 棋盘角点检测成功，共检测到 {len(corners_refined)} 个角点。")
else:
    print("❌ 未检测到棋盘角点，请检查 pattern_size 是否与图像匹配。")

# === 显示结果 ===
cv2.imshow("Chessboard Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
