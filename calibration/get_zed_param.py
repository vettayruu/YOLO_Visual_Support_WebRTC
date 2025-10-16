import pyzed.sl as sl

# 初始化 ZED
zed = sl.Camera()
init_params = sl.InitParameters()
if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
    print("无法打开相机")
    exit()

# 获取相机信息
cam_info = zed.get_camera_information()

calibration_params = zed.get_camera_information().camera_configuration.calibration_parameters

# 左目内参
fx = calibration_params.left_cam.fx
fy = calibration_params.left_cam.fy
cx = calibration_params.left_cam.cx
cy = calibration_params.left_cam.cy

# 左目畸变系数
dist = calibration_params.left_cam.disto  # [k1,k2,p1,p2,k3]

# 左右目基线
tx = calibration_params.stereo_transform.get_translation().get()[0]

# 左目水平视场角
h_fov = calibration_params.left_cam.h_fov

print("left cam parameters:")
print("fx, fy:", fx, fy)
print("cx, cy:", cx, cy)
print("dist:", dist)
print("tx:", tx)
print("h_fov:", h_fov)

# 右目内参
fx = calibration_params.right_cam.fx
fy = calibration_params.right_cam.fy
cx = calibration_params.right_cam.cx
cy = calibration_params.right_cam.cy

# 左目畸变系数
dist = calibration_params.right_cam.disto  # [k1,k2,p1,p2,k3]

# 左右目基线
tx = calibration_params.stereo_transform.get_translation().get()[1]

# 左目水平视场角
h_fov = calibration_params.right_cam.h_fov

print("right cam parameters:")
print("fx, fy:", fx, fy)
print("cx, cy:", cx, cy)
print("dist:", dist)
print("tx:", tx)
print("h_fov:", h_fov)

zed.close()
