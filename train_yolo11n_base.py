from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("ultralytics/cfg/models/11/yolo11.yaml").load("yolo11n.pt")

    model.train(
        data="data/data.yaml",
        epochs=200,
        imgsz=640,
        batch=8,
        workers=0,
        device=0,

        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        cos_lr=True,
        weight_decay=0.0005,

        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,

        degrees=0.0,
        translate=0.0,
        scale=0.0,
        shear=0.0,
        perspective=0.0,

        hsv_h=0.015,
        hsv_s=0.2,
        hsv_v=0.2,
        fliplr=0.5,
        flipud=0.0,

        patience=50,
        seed=42,
        deterministic=True,

        name="neu_det_yolo11n_base"
    )