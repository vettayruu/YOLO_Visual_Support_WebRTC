## Visual Support by YOLO Segmentation

The objective of this module is to provide intuitive visual feedback for the operator.
YOLO is employed to segment the robot’s tips and generate corresponding mask images.
All trained classes are segmented and grouped according to their categories.
The pose of each detected object is then estimated based on the inertia of its mask image.

<div align="center">
  <img src="./yolo_left_1760065195046.jpg" alt="System Architecture" width="1000"/>
  <p><em>Figure 1: Visual Assistane by YOLO's Segmentation. The operator is only required to align the line connecting tip_1 and tip_2 with the orientation of the pack.</em></p>
</div>
