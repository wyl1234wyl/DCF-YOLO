# make_failure_case_fig.py
# Generate failure-case analysis figure for NEU-DET
# Output: runs/detect/failure_cases/failure_case_analysis_300dpi.png / .pdf

import argparse
import random
from pathlib import Path

import cv2
import yaml
import matplotlib
matplotlib.use('Agg')  # 强制使用非交互式后端，解决 PyCharm 绘图报错
import matplotlib.pyplot as plt
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]


def norm_name(x):
    return str(x).lower().replace(" ", "_").replace("-", "_")


def find_yaml_auto(root):
    candidates = []
    for p in root.rglob("*.yaml"):
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if isinstance(data, dict) and "names" in data:
                score = 0
                text = str(p).lower()
                if "neu" in text:
                    score += 5
                if "data.yaml" in text:
                    score += 3
                names = data.get("names")
                if isinstance(names, dict):
                    names_list = [str(v).lower() for v in names.values()]
                elif isinstance(names, list):
                    names_list = [str(v).lower() for v in names]
                else:
                    names_list = []
                if any("crazing" in n for n in names_list):
                    score += 10
                candidates.append((score, p))
        except Exception:
            pass

    candidates = sorted(candidates, key=lambda x: x[0], reverse=True)
    if not candidates:
        raise FileNotFoundError("没有找到 data.yaml，请手动指定 --data 路径。")

    print("\nAuto found yaml candidates:")
    for s, p in candidates[:10]:
        print(f"score={s}  {p}")

    print(f"\nUse data yaml: {candidates[0][1]}\n")
    return candidates[0][1]


def find_weight_auto(root, mode="yolo"):
    pts = list(root.rglob("best.pt"))
    if not pts:
        raise FileNotFoundError("没有找到任何 best.pt 权重文件。")

    scored = []
    for p in pts:
        text = str(p).lower()
        score = 0

        if mode == "yolo":
            if "yolo11" in text or "yolov11" in text:
                score += 10
            if "neu" in text:
                score += 5
            if "dense" in text or "dcf" in text or "daf" in text or "cbam" in text:
                score -= 10

        if mode == "dcf":
            if "densecorr" in text:
                score += 12
            if "dcf" in text:
                score += 12
            if "daf" in text:
                score += 10
            if "dense" in text or "cbam" in text:
                score += 5
            if "neu" in text:
                score += 3

        scored.append((score, p))

    scored = sorted(scored, key=lambda x: x[0], reverse=True)

    print(f"\nAuto found {mode} weight candidates:")
    for s, p in scored[:10]:
        print(f"score={s}  {p}")

    print(f"\nUse {mode} weight: {scored[0][1]}\n")
    return scored[0][1]


def read_class_names(data_yaml):
    data_yaml = Path(data_yaml).resolve()
    with open(data_yaml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    names = data.get("names", None)
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names.keys(), key=lambda x: int(x))]
    elif isinstance(names, list):
        names = names
    else:
        names = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]

    return names, data


def collect_images_from_dir(image_dir):
    image_dir = Path(image_dir).resolve()
    imgs = []
    if not image_dir.exists():
        return imgs

    for ext in IMG_EXTS:
        imgs.extend(image_dir.rglob(f"*{ext}"))
        imgs.extend(image_dir.rglob(f"*{ext.upper()}"))

    return sorted(list(set(imgs)))


def image_to_label_path(img_path, labels_root=None, images_root=None):
    img_path = Path(img_path).resolve()

    if labels_root is not None and images_root is not None:
        try:
            rel = img_path.relative_to(Path(images_root).resolve())
            return Path(labels_root).resolve() / rel.with_suffix(".txt")
        except Exception:
            pass

    parts = list(img_path.parts)
    if "images" in parts:
        idx = parts.index("images")
        parts[idx] = "labels"
        return Path(*parts).with_suffix(".txt")

    return img_path.with_suffix(".txt")


def find_images_from_yaml(data_yaml, data):
    data_yaml = Path(data_yaml).resolve()

    dataset_root = data.get("path", None)
    if dataset_root is None:
        dataset_root = data_yaml.parent
    else:
        dataset_root = Path(dataset_root)
        if not dataset_root.is_absolute():
            dataset_root = (data_yaml.parent / dataset_root).resolve()

    split = data.get("test", None) or data.get("val", None)
    if split is None:
        return []

    split_paths = split if isinstance(split, list) else [split]
    all_imgs = []

    print("\nChecking paths from data.yaml:")
    for sp in split_paths:
        sp = Path(sp)
        if not sp.is_absolute():
            sp = (dataset_root / sp).resolve()

        print("yaml image path:", sp, "exists:", sp.exists())

        if sp.is_dir():
            all_imgs.extend(collect_images_from_dir(sp))

        elif sp.is_file() and sp.suffix.lower() == ".txt":
            with open(sp, "r", encoding="utf-8") as f:
                for line in f:
                    ip = Path(line.strip())
                    if not ip.is_absolute():
                        ip = (sp.parent / ip).resolve()
                    if ip.exists() and ip.suffix.lower() in IMG_EXTS:
                        all_imgs.append(ip)

    return sorted(list(set(all_imgs)))


def find_images_auto(root):
    print("\nData.yaml path invalid or empty. Auto-searching image folders...")

    candidate_dirs = []
    for d in root.rglob("*"):
        if not d.is_dir():
            continue

        low = str(d).lower()
        if "images" in low and ("val" in low or "test" in low or "valid" in low):
            imgs = collect_images_from_dir(d)
            if len(imgs) > 0:
                score = len(imgs)
                if "neu" in low:
                    score += 1000
                candidate_dirs.append((score, d, len(imgs)))

    candidate_dirs = sorted(candidate_dirs, key=lambda x: x[0], reverse=True)

    if not candidate_dirs:
        # fallback: search any image folders
        for d in root.rglob("*"):
            if d.is_dir():
                imgs = collect_images_from_dir(d)
                if len(imgs) >= 20:
                    candidate_dirs.append((len(imgs), d, len(imgs)))
        candidate_dirs = sorted(candidate_dirs, key=lambda x: x[0], reverse=True)

    if not candidate_dirs:
        raise RuntimeError("自动搜索也没有找到图片目录。请手动指定 --images。")

    print("\nAuto found image folder candidates:")
    for s, d, n in candidate_dirs[:10]:
        print(f"score={s}  images={n}  {d}")

    use_dir = candidate_dirs[0][1]
    imgs = collect_images_from_dir(use_dir)

    print(f"\nUse image folder: {use_dir}")
    print(f"Images found: {len(imgs)}\n")

    return imgs, use_dir


def load_gt(label_path, w, h):
    label_path = Path(label_path)
    boxes = []

    if not label_path.exists():
        return boxes

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            items = line.strip().split()
            if len(items) < 5:
                continue

            cls = int(float(items[0]))
            xc, yc, bw, bh = map(float, items[1:5])

            x1 = (xc - bw / 2) * w
            y1 = (yc - bh / 2) * h
            x2 = (xc + bw / 2) * w
            y2 = (yc + bh / 2) * h

            boxes.append({"cls": cls, "conf": 1.0, "box": [x1, y1, x2, y2]})

    return boxes


def predict_boxes(model, img_path, imgsz, conf):
    result = model.predict(source=str(img_path), imgsz=imgsz, conf=conf, verbose=False)[0]

    preds = []
    if result.boxes is None:
        return preds

    xyxy = result.boxes.xyxy.cpu().numpy()
    cls = result.boxes.cls.cpu().numpy()
    score = result.boxes.conf.cpu().numpy()

    for b, c, s in zip(xyxy, cls, score):
        preds.append({"cls": int(c), "conf": float(s), "box": b.tolist()})

    return preds


def iou_xyxy(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])

    return inter / (area_a + area_b - inter + 1e-6)


def match_gt(gt, preds, iou_thr):
    best = None
    best_iou = 0.0

    for p in preds:
        if p["cls"] != gt["cls"]:
            continue
        iou = iou_xyxy(gt["box"], p["box"])
        if iou > best_iou:
            best_iou = iou
            best = p

    if best is not None and best_iou >= iou_thr:
        return best
    return None


def is_false_positive(pred, gts, iou_thr):
    for gt in gts:
        if pred["cls"] == gt["cls"] and iou_xyxy(pred["box"], gt["box"]) >= iou_thr:
            return False
    return True


def draw_boxes(img_rgb, boxes, names, mode):
    out = img_rgb.copy()

    for item in boxes:
        cls = int(item["cls"])
        conf = float(item.get("conf", 1.0))
        x1, y1, x2, y2 = map(int, item["box"])

        if mode == "gt":
            color = (255, 255, 255)
            label = f"GT: {names[cls]}" if cls < len(names) else f"GT: {cls}"
        else:
            color = (255, 0, 255)
            label = f"{names[cls]} {conf:.2f}" if cls < len(names) else f"{cls} {conf:.2f}"

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            out,
            label,
            (x1, max(18, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA
        )

    return out


def find_failure_cases(image_paths, names, yolo_model, dcf_model, args, images_root=None, labels_root=None):
    random.seed(args.seed)
    paths = list(image_paths)
    random.shuffle(paths)

    selected = {
        "Missed detection": None,
        "False positive": None,
        "Low confidence": None
    }
    cache = {}

    checked = 0
    with_gt = 0

    for img_path in paths:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue

        checked += 1
        h, w = img_bgr.shape[:2]
        label_path = image_to_label_path(img_path, labels_root=labels_root, images_root=images_root)
        gts = load_gt(label_path, w, h)

        if len(gts) == 0:
            continue

        with_gt += 1

        yolo_preds = predict_boxes(yolo_model, img_path, args.imgsz, args.conf)
        dcf_preds = predict_boxes(dcf_model, img_path, args.imgsz, args.conf)

        cache[img_path] = (gts, yolo_preds, dcf_preds)

        if selected["Missed detection"] is None:
            for gt in gts:
                cls_name = norm_name(names[gt["cls"]]) if gt["cls"] < len(names) else ""
                if cls_name in ["crazing", "pitted_surface", "scratches"]:
                    if match_gt(gt, dcf_preds, args.iou) is None:
                        selected["Missed detection"] = img_path
                        break

        if selected["False positive"] is None:
            fps = [p for p in dcf_preds if is_false_positive(p, gts, args.iou)]
            if len(fps) > 0:
                selected["False positive"] = img_path

        if selected["Low confidence"] is None:
            for gt in gts:
                p = match_gt(gt, dcf_preds, args.iou)
                if p is not None and args.low_min <= p["conf"] < args.low_max:
                    selected["Low confidence"] = img_path
                    break

        if all(v is not None for v in selected.values()):
            break

    print(f"\nImages checked: {checked}")
    print(f"Images with labels: {with_gt}")

    print("\nSelected failure cases:")
    for k, v in selected.items():
        print(k, ":", v)

    return selected, cache


def make_figure(selected, cache, names, save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    rows = ["Missed detection", "False positive", "Low confidence"]
    cols = ["Original / GT", "YOLOv11n", "DCF-YOLO"]

    fig, axes = plt.subplots(3, 3, figsize=(9.2, 7.0), dpi=300)

    for r, row_name in enumerate(rows):
        img_path = selected.get(row_name)

        if img_path is None or img_path not in cache:
            for c in range(3):
                axes[r, c].axis("off")
                axes[r, c].text(0.5, 0.5, "No case found", ha="center", va="center")
            continue

        img_bgr = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        gts, yolo_preds, dcf_preds = cache[img_path]

        show_imgs = [
            draw_boxes(img_rgb, gts, names, "gt"),
            draw_boxes(img_rgb, yolo_preds, names, "pred"),
            draw_boxes(img_rgb, dcf_preds, names, "pred")
        ]

        for c in range(3):
            axes[r, c].imshow(show_imgs[c])
            axes[r, c].axis("off")

            if r == 0:
                axes[r, c].set_title(cols[c], fontsize=12, fontweight="bold")

            if c == 0:
                axes[r, c].set_ylabel(row_name, fontsize=11, fontweight="bold")

    plt.tight_layout()

    png_path = save_dir / "failure_case_analysis_300dpi.png"
    pdf_path = save_dir / "failure_case_analysis_300dpi.pdf"

    plt.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.savefig(pdf_path, bbox_inches="tight", pad_inches=0.05)
    plt.close()

    print("\nSaved:")
    print(png_path)
    print(pdf_path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", type=str, default="", help="Path to data.yaml. Empty means auto-search.")
    parser.add_argument("--images", type=str, default="", help="Path to images val/test folder. Optional.")
    parser.add_argument("--labels", type=str, default="", help="Path to labels val/test folder. Optional.")

    parser.add_argument("--yolo", type=str, default="", help="Path to YOLOv11n best.pt. Empty means auto-search.")
    parser.add_argument("--dcf", type=str, default="", help="Path to DCF-YOLO best.pt. Empty means auto-search.")
    parser.add_argument("--save", type=str, default=str(ROOT / "runs/detect/failure_cases"))

    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--low-min", type=float, default=0.25)
    parser.add_argument("--low-max", type=float, default=0.70)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    data_yaml = Path(args.data).resolve() if args.data else find_yaml_auto(ROOT)
    names, data = read_class_names(data_yaml)

    if args.images:
        image_paths = collect_images_from_dir(args.images)
        images_root = Path(args.images).resolve()
        labels_root = Path(args.labels).resolve() if args.labels else None
        print(f"\nUse manually specified images: {images_root}")
        print(f"Images found: {len(image_paths)}")
        if labels_root:
            print(f"Use manually specified labels: {labels_root}")
    else:
        image_paths = find_images_from_yaml(data_yaml, data)
        images_root = None
        labels_root = None

        if len(image_paths) == 0:
            image_paths, images_root = find_images_auto(ROOT)
            labels_root = None

    if len(image_paths) == 0:
        raise RuntimeError("没有找到任何图片。请用 --images 手动指定图片文件夹。")

    yolo_weight = Path(args.yolo).resolve() if args.yolo else find_weight_auto(ROOT, "yolo")
    dcf_weight = Path(args.dcf).resolve() if args.dcf else find_weight_auto(ROOT, "dcf")

    print("\nClass names:", names)
    print("Total images:", len(image_paths))

    print("\nLoading YOLOv11n...")
    yolo_model = YOLO(str(yolo_weight))

    print("Loading DCF-YOLO...")
    dcf_model = YOLO(str(dcf_weight))

    print("\nSearching failure cases...")
    selected, cache = find_failure_cases(
        image_paths=image_paths,
        names=names,
        yolo_model=yolo_model,
        dcf_model=dcf_model,
        args=args,
        images_root=images_root,
        labels_root=labels_root
    )

    print("\nMaking figure...")
    make_figure(selected, cache, names, args.save)


if __name__ == "__main__":
    main()