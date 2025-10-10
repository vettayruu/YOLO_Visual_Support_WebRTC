## Visual Support by YOLO Segmentation

The objective of this module is to provide intuitive visual feedback for the operator.
YOLO is employed to segment the robot’s tips and generate corresponding mask images.
All trained classes are segmented and grouped according to their categories.
The pose of each detected object is then estimated based on the inertia of its mask image.

<div align="center">
  <img src="./yolo_left_1760065195046.jpg" alt="System Architecture" width="1000"/>
  <p><em>Figure 1: Visual Assistane by YOLO's Segmentation. The operator is only required to align the line connecting tip_1 and tip_2 with the orientation of the pack.</em></p>
</div>

# YOLO Training Guide (Segmentation & Pose)

This document provides a step-by-step guide to prepare, convert, and train YOLO models — focusing on segmentation tasks for robotic arm and gripper detection.

---

## 1️⃣ Prepare Dataset

### Annotation with Labelme
Use **Labelme** for dataset annotation.

**Installation:**
```bash
pip install labelme
```

**Run Labelme:**
```bash
labelme
```
Labelme supports SAM (Segment Anything Model) integration —
use AI-Polygon mode for fast and accurate segmentation.

## 2️⃣ Convert JSON to YOLO Format
After labeling, convert `.json` files to YOLO-style `.txt` annotations using a conversion script.

**Run conversion:**

```bash
python json2yolo.py
```

Notes:

- The class names defined in `data.yaml` must match those used in `json2yolo.py`.
- Each image should have a corresponding `.txt` label file with the same filename.

## 3️⃣ Dataset Directory Structure
Organize your dataset as follows:

```bash
dataset/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

- images/ — contains training and validation images.
- labels/ — contains YOLO-format label files (one per image).

## 4️⃣ Configure data.yaml
`data.yaml` defines dataset paths and class information.

Example:
```bash
train: ./images/train
val: ./images/val

nc: 2   # number of classes
names: ['arm', 'gripper']
```

Make sure that the class order is consistent with your annotation script.

## 5️⃣ Train YOLO Segmentation Model
Use your training script (for example, yolo_model_train_seg.py):

```bash
python yolo_model_train_seg.py
```

After training, the model weights will be saved at:
```bash
runs/segment/exp_yolo11_seg/weights/best.pt
```

## Recommendation
- Start with Segmentation Model (YOLO-Seg) for higher accuracy and easier training.
- The Pose Model (YOLO-Pose) can be considered for advanced applications once you have a well-labeled dataset.
