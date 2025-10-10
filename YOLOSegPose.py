from polars import arctan2
from torch.cuda import device
from ultralytics import YOLO
import numpy as np
import cv2
import time

class YOLOSegPose:
    def __init__(self, model_path: str):
        """
        初始化 YOLO 分割模型
        """
        self.model = YOLO(model_path, verbose=False)

        self.length_tip = 100
        self.length_object = 100

        self.img_width = 1280
        self.img_height = 720

        self.left_arm_box = [0, 0, 0, 0] # x1, y1, x2, y2
        self.right_arm_box = [0, 0, 0, 0]

    def infer(self, img):
        """
        推理图像，返回每个 mask 的类别、中心和主方向
        :param img: np.ndarray 或 图像路径
        :return: List[dict] [{'class': name, 'cx': float, 'cy': float, 'theta': float}, ...]
        """
        # 如果输入是路径，读取图像
        if isinstance(img, str):
            img = cv2.imread(img)

        # self.img_width = int(img.shape[1])
        # self.img_height = int(img.shape[0])

        # results = self.model(img)[0]

        results = self.model.predict(
            source=img,  # 可以是numpy数组、路径或图像对象
            conf=0.5,  # 置信度阈值（默认0.25）
            iou=0.6,  # IoU阈值（默认0.7）
            device = 0, # cuda
            verbose=False,
        )[0]

        # print(self.model.device)

        output = []
        if results.masks is None:
            print("⚠️ No detections found.")
            return []

        for i, (box, mask) in enumerate(zip(results.boxes, results.masks.data)):
            mask = mask.cpu().numpy().astype(np.uint8)

            mask = cv2.GaussianBlur(mask, (7, 7), 0)

            # resize mask 到原图大小
            mask = cv2.resize(mask, (self.img_width, self.img_height), interpolation=cv2.INTER_NEAREST)

            # 类别和置信度
            cls_id = int(results.boxes.cls[i])
            cls_name = results.names[cls_id]

            # bounding box
            bbox = box.xyxy[0].tolist()

            # 计算质心和方向
            M = cv2.moments(mask)
            if M["m00"] == 0:  # 避免除0
                continue
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            theta = 0.5 * np.arctan2(2 * M["mu11"], M["mu20"] - M["mu02"])
            theta_deg = np.degrees(theta)

            group_id = ''
            if cls_name == 'piper':
                if cx < self.img_width/2:
                    group_id = 'left'
                    self.left_arm_box = bbox
                elif cx >= self.img_width/2:
                    group_id = 'right'
                    self.right_arm_box = bbox

            else:
                group_id = 'object'

            output.append({
                'group': group_id,
                'class': cls_name,
                'cx': cx,
                'cy': cy,
                # 'bbox': bbox,
                'theta': theta_deg
            })

        lx1, ly1, lx2, ly2 = self.left_arm_box
        rx1, ry1, rx2, ry2 = self.right_arm_box

        for item in output:
            if item['class'] in ('tip', 'tip_1', 'tip_2'):
                cx = item.get('cx', 0)
                if lx1 < lx2 and lx1 <= cx <= lx2:
                    item['group'] = 'left'
                elif rx1 < rx2 and rx1 <= cx <= rx2:
                    item['group'] = 'right'
                # else:
                #     item['group'] = item.get('group', '')

        return output

    def draw_pose(self, img, cx, cy, theta, item_class, length):
        cv2.circle(img, (int(cx), int(cy)), 5, (255, 0, 0), -1)

        # y direction
        x1 = int(cx + length/2 * np.cos(theta + np.pi / 2))
        y1 = int(cy + length/2 * np.sin(theta + np.pi / 2))
        cv2.line(img, (int(cx), int(cy)), (x1, y1), (0, 255, 0), 2)

        # x direction
        x2 = int(cx + length * np.cos(theta + np.pi))
        y2 = int(cy + length * np.sin(theta + np.pi))
        cv2.line(img, (int(cx), int(cy)), (x2, y2), (0, 0, 255), 2)

        # 绘制类别
        cv2.putText(img, item_class, (int(cx), int(cy) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    def visualize(self, img):
        """
        可视化 mask 主方向
        """
        if isinstance(img, str):
            img = cv2.imread(img)

        output = self.infer(img)

        groups = {}
        for item in output:
            g = item.get('group', '') or 'ungrouped'
            groups.setdefault(g, []).append(item)

        for g, items in groups.items():
            # print('group:', g)
            # print('items:', items)

            if g == 'left':
                cx_tip_1 = cy_tip_1 = theta_tip_1 = None
                cx_tip_2 = cy_tip_2 = theta_tip_2 = None

                for item in items:
                    if item['class'] != 'piper':
                        cx, cy, theta = item['cx'], item['cy'], np.radians(item['theta'])
                        self.draw_pose(img, cx, cy, theta+np.pi, item['class'], self.length_tip)

                    if item['class'] == 'tip_1':
                        cx_tip_1, cy_tip_1 = item['cx'], item['cy']
                        theta_tip_1 = np.radians(item['theta'])

                    elif item['class'] == 'tip_2':
                        cx_tip_2, cy_tip_2 = item['cx'], item['cy']
                        theta_tip_2 = np.radians(item['theta'])

                    if cx_tip_1 is not None and cx_tip_2 is not None:
                        tx_tip_1 = int(cx_tip_1 + self.length_tip * np.cos(theta_tip_1))
                        ty_tip_1 = int(cy_tip_1 + self.length_tip* np.sin(theta_tip_1))

                        tx_tip_2 = int(cx_tip_2 + self.length_tip * np.cos(theta_tip_2))
                        ty_tip_2 = int(cy_tip_2 + self.length_tip * np.sin(theta_tip_2))

                        cv2.line(img, (tx_tip_1, ty_tip_1), (tx_tip_2, ty_tip_2),
                                 (0, 0, 255), 2)

            elif g == 'right':
                cx_tip_1 = cy_tip_1 = theta_tip_1 = None
                cx_tip_2 = cy_tip_2 = theta_tip_2 = None

                for item in items:
                    if item['class'] != 'piper':
                        cx, cy, theta = item['cx'], item['cy'], np.radians(item['theta'])
                        self.draw_pose(img, cx, cy, theta, item['class'], self.length_tip)

                    if item['class'] == 'tip_1':
                        cx_tip_1, cy_tip_1 = item['cx'], item['cy']
                        theta_tip_1 = np.radians(item['theta'])

                    elif item['class'] == 'tip_2':
                        cx_tip_2, cy_tip_2 = item['cx'], item['cy']
                        theta_tip_2 = np.radians(item['theta'])

                    if cx_tip_1 is not None and cx_tip_2 is not None:
                        tx_tip_1 = int(cx_tip_1 + self.length_tip * np.cos(theta_tip_1 + np.pi))
                        ty_tip_1 = int(cy_tip_1 + self.length_tip * np.sin(theta_tip_1 + np.pi))

                        tx_tip_2 = int(cx_tip_2 + self.length_tip * np.cos(theta_tip_2 + np.pi))
                        ty_tip_2 = int(cy_tip_2 + self.length_tip * np.sin(theta_tip_2 + np.pi))

                        cv2.line(img, (tx_tip_1, ty_tip_1), (tx_tip_2, ty_tip_2),
                                 (0, 0, 255), 2)

            else:
                for item in items:
                    cx, cy, theta = item['cx'], item['cy'], np.radians(item['theta'])
                    self.draw_pose(img, cx, cy, theta, item['class'], self.length_object)

        # cv2.imshow("YOLO Seg Pose", img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

