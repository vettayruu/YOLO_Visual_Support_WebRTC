# train_yolo11_seg.py
from ultralytics import YOLO

def main():
    # 1️⃣ 载入预训练模型（YOLOv11 分割模型）
    model = YOLO("yolo11n-seg.pt")  # 可换为 yolo11m-seg.pt / yolo11l-seg.pt 等

    # 2️⃣ 训练配置
    model.train(
        data="data.yaml",       # 你的数据配置文件路径
        epochs=50,             # 训练轮数
        imgsz=640,              # 图片尺寸
        batch=8,                # 每批次大小（显存不够可以改小）
        project="runs/segment", # 输出目录
        name="exp_yolo11_seg",  # 实验名称
        device=0,               # GPU设备编号（用CPU则设为 'cpu'）
        lr0=0.001,              # 初始学习率
        patience=30,            # 无提升自动停止
        augment=True,           # 开启数据增强
        cache=False,            # 是否缓存图片（首次训练建议False）
    )

    # 3️⃣ 训练完成后自动验证
    model.val(data="data.yaml")

    print("\n✅ 训练完成！结果已保存到 runs/segment/exp_yolo11_seg")

if __name__ == "__main__":
    main()
