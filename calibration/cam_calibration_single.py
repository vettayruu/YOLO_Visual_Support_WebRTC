import cv2
import numpy as np

# 1️⃣ 读取图像并转灰度
img_path = "./left/zed_left_1760576521300.jpg"
img = cv2.imread(img_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2️⃣ 检测角点
pattern_size = (9, 6)
ret, corners = cv2.findChessboardCorners(gray, pattern_size)

if ret:
    # 3️⃣ 亚像素角点优化（让角点位置更精确）
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners_subpix = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)

    # 4️⃣ 绘制辅助线（棋盘）
    img_with_lines = img.copy()
    cv2.drawChessboardCorners(img_with_lines, pattern_size, corners_subpix, ret)

    # 5️⃣ 临时单张标定
    objp = np.zeros((pattern_size[0]*pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

    obj_points = [objp]
    img_points = [corners_subpix]

    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, gray.shape[::-1], None, None
    )

    print("相机内参矩阵 K：\n", K)
    print("畸变系数 dist：\n", dist)

    # 6️⃣ 去畸变（保持原始尺寸，不裁剪）
    h, w = gray.shape[:2]
    newcameramtx, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 0.6, (w, h))
    undistorted = cv2.undistort(img, K, dist, None, newcameramtx)

    # 7️⃣ 显示结果
    cv2.imshow("Detected Chessboard (with lines)", img_with_lines)
    cv2.imshow("Undistorted (full view)", undistorted)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("未检测到棋盘角点！")
