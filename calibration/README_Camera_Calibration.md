# 📸 Camera Distortion Correction Workflow

Camera lenses often introduce **geometric distortions** that reduce image accuracy — an important issue in applications such as **3D reconstruction**, **stereo vision**, and **robotics**. 
This document summarizes the workflow for **camera distortion calibration and correction**.

---

## 1. Understanding Camera Distortion
Real-world camera lenses deviate from the ideal pinhole camera model due to optical imperfections. These distortions are typically categorized as:

### Radial Distortion

Radial distortion is caused by the **curvature of the lens elements**, resulting in either:

- **Barrel distortion** → image bulges outward (lines curve away from the center)
- **Pincushion distortion** → image curves inward (lines bend toward the center)

The correction model can be expressed as:

$$
x_{corr} = x (1 + k_1 r^2 + k_2 r^4 + k_3 r^6)
$$

$$
y_{corr} = y (1 + k_1 r^2 + k_2 r^4 + k_3 r^6)
$$

### Tangential Distortion

Tangential distortion occurs when the **lens and image sensor** are not perfectly parallel,  
causing image points to shift diagonally.

The correction model is:

$$
x_{corr} = x + 2p_1xy + p_2(r^2 + 2x^2)
$$

$$
y_{corr} = y + p_1(r^2 + 2y^2) + 2p_2xy
$$

where  

$$r^2 = x^2 + y^2$$  

and the parameters are:  
- $$k_1, k_2, k_3$$: **radial distortion coefficients**  
- $$p_1, p_2$$: **tangential distortion coefficients**

---

## 2. ZED Mini Undistortion Model

The **ZED Mini** camera follows the **pinhole camera model**,  
which relates 3D world coordinates to 2D image coordinates via the **intrinsic matrix** $$K$$:
  
$$
K =
\begin{bmatrix}
f_x & 0   & c_x \\
0   & f_y & c_y \\
0   & 0   & 1
\end{bmatrix}
$$

Here:  
- \(f_x, f_y\) are the **focal lengths** in pixels  
- \(c_x, c_y\) are the **principal point offsets**

and distortion parameter `dist` is defined as:

$$
dist = [k_1, k_2, k_3, p_1, p_2]
$$


---



  
  
