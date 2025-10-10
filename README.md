## Visual Support by YOLO Segmentation

The objective is to provide visual feedback to the operator.

YOLO is used to segment the robot's tip and generate corresponding mask images.

All trained classes are segmented and grouped according to their categories.

The pose of each object is estimated based on the inertia of its mask image.

<div align="center">
  <img src="./MQTT_Control_Framework.svg" alt="System Architecture" width="1000"/>
  <p><em>Figure 1: System Architecture of PiPER Control via MetaworkMQTT</em></p>
</div>
