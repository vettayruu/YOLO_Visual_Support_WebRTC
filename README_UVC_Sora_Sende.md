# Camera Undistortion for ZED Mini (UVC Mode)

ZED Mini can be used as a **UVC (USB Video Class)** camera, allowing direct access through **OpenCV**.  
However, OpenCV only provides **raw frames**, which means the images include **lens distortion**.

To correct this distortion, there are two main approaches.

---

## 1. Using `a-curvedimage` in A-Frame (VR Rendering)

This method doesn’t require explicit image undistortion.  
Instead, it maps the raw image onto a **curved surface** in VR, creating a wide-FOV undistorted appearance.

```html
<a-curvedimage
  id="left-curved"
  height="7.0"
  radius="5.7"
  theta-length="120"
  position="0.2 1.6 -1.0"
  rotation="0 -115 0"
  scale="-1 1 1"
  stereo-curvedvideo="eye: left; videoId: leftVideo">
</a-curvedimage>

<a-curvedimage
  id="right-curved"
  height="7.0"
  radius="5.7"
  theta-length="120"
  position="0.2 1.6 -1.0"
  rotation="0 -120.3 0"
  scale="-1 1 1"
  stereo-curvedvideo="eye: right; videoId: rightVideo">
</a-curvedimage>
