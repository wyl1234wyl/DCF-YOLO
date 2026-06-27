from ultralytics import YOLO
import torch

if __name__ == '__main__':
    device = 0 if torch.cuda.is_available() else 'cpu'

    # 先按你自定义yaml搭模型，再加载原版预训练权重
    #model = YOLO('ultralytics/cfg/models/v8/yolov8n_cbam.yaml').load('yolov8n.pt')
    # model = YOLO('ultralytics/cfg/models/v8/yolov8n_denseeaf.yaml').load('yolov8n.pt')
    # model = YOLO('ultralytics/cfg/models/v8/yolov8.yaml').load('yolov8n.pt')
    # model = YOLO('ultralytics/cfg/models/11/yolo11.yaml')
    # model = YOLO('yolov8s.pt')
    # model = YOLO("ultralytics/cfg/models/11/yolo11.yaml").load("yolo11s.pt")
    model = YOLO("ultralytics/cfg/models/11/yolo11n_densecbam_dbcleaf_p2.yaml").load("yolo11n.pt")
    # model = YOLO("ultralytics/cfg/models/11/yolo11n_densecbam_p4_dbcleaf_p3p4.yaml").load("yolo11s.pt")
    model.info()

    model.train(
        data='data/data.yaml',
        epochs=200,
        imgsz=640,
        batch=16,
        workers=0,
        device=device,
        optimizer='AdamW',
        lr0=0.001,
        # name='neu_det_yolov8',
        name='yolo11n_densecbam_dbcleaf_p2_m',
        mosaic=True , # 开启 Mosaic
        patience=30,  # 早停：30轮没提升就自动停止

    )