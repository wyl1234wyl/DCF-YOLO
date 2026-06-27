import shutil
from pathlib import Path

# 原训练集
root = Path("data/NEU-DET")
src_img_dir = root / "train/images"
src_lbl_dir = root / "train/labels"

# 过采样后的训练集
dst_img_dir = root / "train_os/images"
dst_lbl_dir = root / "train_os/labels"

dst_img_dir.mkdir(parents=True, exist_ok=True)
dst_lbl_dir.mkdir(parents=True, exist_ok=True)

# 类别编号，按照你的 data.yaml：
# 0 crazing
# 1 inclusion
# 2 patches
# 3 pitted_surface
# 4 rolled-in_scale
# 5 scratches
target_classes = {0, 4}

# 先复制原始训练集
for img_path in src_img_dir.glob("*.*"):
    if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp"]:
        continue

    lbl_path = src_lbl_dir / f"{img_path.stem}.txt"
    if not lbl_path.exists():
        continue

    shutil.copy2(img_path, dst_img_dir / img_path.name)
    shutil.copy2(lbl_path, dst_lbl_dir / lbl_path.name)

    # 判断是否包含小目标/低精度类别
    labels = lbl_path.read_text(encoding="utf-8").strip().splitlines()
    cls_ids = set()
    for line in labels:
        if line.strip():
            cls_ids.add(int(float(line.split()[0])))

    # 如果是低精度类别，额外复制 2 份
    if cls_ids & target_classes:
        for k in range(2):
            new_img_name = f"{img_path.stem}_os{k}{img_path.suffix}"
            new_lbl_name = f"{img_path.stem}_os{k}.txt"

            shutil.copy2(img_path, dst_img_dir / new_img_name)
            shutil.copy2(lbl_path, dst_lbl_dir / new_lbl_name)

print("过采样完成！")
print("新训练图片数量:", len(list(dst_img_dir.glob('*.*'))))
print("新训练标签数量:", len(list(dst_lbl_dir.glob('*.txt'))))