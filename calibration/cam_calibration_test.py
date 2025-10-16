# 相机内参矩阵 K:
#  [[788.11940229   0.         660.32698318]
#  [  0.         787.09939768 346.13751838]
#  [  0.           0.           1.        ]]
# 畸变系数 dist:
#  [[-3.61648028e-01  2.26732575e-01 -1.19047650e-03  2.61003820e-04
#   -9.78096249e-02]]

import cv2
import numpy as np

# ==== 1️⃣ 读取图像 ====
img_path = "./left/zed_left_1760572454938.jpg"  # 替换成你的测试图
img = cv2.imread(img_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ==== 2️⃣ 已知相机内参矩阵和畸变系数 ====
K = np.array([[788.11940229, 0., 660.32698318],
              [0., 787.09939768, 346.13751838],
              [0., 0., 1.]], dtype=np.float32)

dist = np.array([[-0.361648028, 0.226732575, -0.0011904765, 0.00026100382, -0.0978096249]], dtype=np.float32)

# ==== 3️⃣ 检测棋盘角点（辅助线可视化） ====
pattern_size = (9, 6)
ret, corners = cv2.findChessboardCorners(gray, pattern_size)

if ret:
    # 亚像素优化
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners_subpix = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)

    # 绘制辅助线
    img_with_lines = img.copy()
    cv2.drawChessboardCorners(img_with_lines, pattern_size, corners_subpix, ret)
else:
    print("未检测到棋盘角点！")
    img_with_lines = img.copy()

# ==== 4️⃣ 去畸变 ====
h, w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 0, (w, h))
undistorted = cv2.undistort(img, K, dist, None, newcameramtx)

print("newcameramtx", newcameramtx)
print("roi", roi)

# ==== 5️⃣ 显示结果 ====
cv2.imshow("Original", img)
cv2.imshow("Chessboard with lines", img_with_lines)
cv2.imshow("Undistorted", undistorted)
cv2.waitKey(0)
cv2.destroyAllWindows()
