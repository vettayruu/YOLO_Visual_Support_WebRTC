import cv2
import numpy as np
import glob

# 棋盘内角点数
pattern_size = (9, 6)

# 棋盘三维点准备 (z=0)
objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

# 存储所有图像的角点
obj_points = []  # 世界坐标
img_points = []  # 图像坐标

# 读取图像列表
images = glob.glob("./right/*.jpg")

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, pattern_size)
    if ret:
        # 亚像素优化
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_subpix = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        obj_points.append(objp)
        img_points.append(corners_subpix)

        # 可视化辅助线
        cv2.drawChessboardCorners(img, pattern_size, corners_subpix, ret)
        cv2.imshow("Corners", img)
        cv2.waitKey(100)
    else:
        print(f"⚠️ 未检测到角点: {fname}")

cv2.destroyAllWindows()

# 相机标定
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    obj_points, img_points, gray.shape[::-1], None, None
)

print("相机内参矩阵 K:\n", K)
print("畸变系数 dist:\n", dist)

# 去畸变示例
img = cv2.imread(images[0])
h, w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 0, (w, h))
undistorted = cv2.undistort(img, K, dist, None, newcameramtx)

cv2.imshow("Original", img)
cv2.imshow("Undistorted", undistorted)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Result left cam
# 相机内参矩阵 K:
#  [[788.11940229   0.         660.32698318]
#  [  0.         787.09939768 346.13751838]
#  [  0.           0.           1.        ]]
# 畸变系数 dist:
#  [[-3.61648028e-01  2.26732575e-01 -1.19047650e-03  2.61003820e-04
#   -9.78096249e-02]]

# Result right cam
# 相机内参矩阵 K:
#  [[788.41415049   0.         655.01692926]
#  [  0.         787.3765135  357.82862631]
#  [  0.           0.           1.        ]]
# 畸变系数 dist:
#  [[-0.3506601   0.18558038 -0.00065609  0.00100313 -0.05786136]]
