# -*- coding: utf-8 -*-
"""
prepare_dataset.py
1. Processes LabelMe JSON files from 'new Dataset' (custom dataset), crops
   the bounding boxes, and sorts into 6 classes.
2. Copies Kaggle dataset from 'DATASET' into 'organic' and 'recyclable' classes.
3. Saves everything to 'datasets/master_dataset/{train,validation,test}/{class}/'
"""
import os, json, shutil, random, base64, io, sys
from pathlib import Path
from PIL import Image

# ── Class mapping: JSON folder name -> Final Class ───────────────────────────
FOLDER_CLASS_MAP = {
    "clothing label":       "clothing",
    "ewaste_label":         "ewaste",
    "glass_label":          "glass",
    "metal_label":          "metal",
    "paper_label":          "paper",
    "plastic_annotated":    "plastic",
    "new labelme(plastic)": "plastic"
}

# ── Class mapping: Raw image folder name -> Final Class ───────────────────────
IMAGE_FOLDER_MAP = {
    "battery":       "ewaste",
    "biological":    "organic",
    "brown-glass":   "glass",
    "cardboard":     "paper",
    "clothes":       "clothing",
    "clothing":      "clothing",
    "ewaste":        "ewaste",
    "glass":         "glass",
    "green-glass":   "glass",
    "metal":         "metal",
    "paper":         "paper",
    "plastic":       "plastic",
    "shoes":         "clothing",
    "trash":         "recyclable",
    "white-glass":   "glass"
}

KAGGLE_MAP = {
    "O": "organic",
    "R": "recyclable"
}

SPLIT = {"train": 0.70, "validation": 0.20, "test": 0.10}

CUSTOM_SOURCE_DIR = Path("new Dataset")
KAGGLE_SOURCE_DIR = Path("DATASET")
DEST_DIR          = Path("datasets/master_dataset")

def crop_and_save(json_path: Path, out_dir: Path) -> Path | None:
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        img_data = data.get("imageData")
        shapes = data.get("shapes", [])
        
        if not img_data or not shapes:
            return None
            
        points = shapes[0].get("points", [])
        if not points:
            return None
            
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        x1, y1 = min(x_coords), min(y_coords)
        x2, y2 = max(x_coords), max(y_coords)
        
        if x2 - x1 < 5 or y2 - y1 < 5:
            return None

        img_bytes = base64.b64decode(img_data)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_cropped = img.crop((x1, y1, x2, y2))
        
        out_path = out_dir / (json_path.stem + ".jpg")
        img_cropped.save(out_path, "JPEG", quality=95)
        return out_path
    except Exception as e:
        return None

def collect_and_crop_custom(folder: Path, class_name: str, dest_temp: Path) -> list[Path]:
    crops = []
    for f in sorted(folder.iterdir()):
        if f.suffix.lower() == ".json":
            out_path = crop_and_save(f, dest_temp)
            if out_path and out_path.exists():
                crops.append(out_path)
    return crops

def collect_raw_images(folder: Path, class_name: str, dest_temp: Path) -> list[Path]:
    """Collects and processes raw image files, converting HEIC/PNG/WEBP/etc to JPG."""
    import pillow_heif
    pillow_heif.register_heif_opener()
    
    imgs = []
    exts = {".jpg", ".jpeg", ".png", ".webp", ".heic"}
    for f in sorted(folder.iterdir()):
        if f.is_file() and f.suffix.lower() in exts:
            out_path = dest_temp / f"{folder.name}_{f.stem}.jpg"
            try:
                img = Image.open(f).convert("RGB")
                img.save(out_path, "JPEG", quality=95)
                imgs.append(out_path)
            except Exception as e:
                print(f"Error copying/converting {f}: {e}")
    return imgs

def collect_kaggle_images(kaggle_dir: Path, folder_name: str, dest_temp: Path) -> list[Path]:
    """Collects images from Kaggle TRAIN and TEST folders and copies to temp."""
    imgs = []
    # Check TRAIN
    train_dir = kaggle_dir / "TRAIN" / folder_name
    test_dir = kaggle_dir / "TEST" / folder_name
    
    for src_dir in [train_dir, test_dir]:
        if not src_dir.exists():
            continue
        for f in src_dir.iterdir():
            if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
                out_path = dest_temp / f"{src_dir.parent.name}_{f.name}"
                shutil.copy2(f, out_path)
                imgs.append(out_path)
    return imgs

def main():
    random.seed(42)
    
    if DEST_DIR.exists():
        shutil.rmtree(DEST_DIR)

    # 6 custom + 2 Kaggle = 8 classes
    classes = sorted(set(FOLDER_CLASS_MAP.values()) | set(KAGGLE_MAP.values()))
    
    for split in SPLIT:
        for cls in classes:
            (DEST_DIR / split / cls).mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Master Dataset Merge (8 Classes)")
    print(f"Dest:   {DEST_DIR.resolve()}")
    print(f"Classes: {classes}")
    print(f"{'='*60}\n")

    class_images: dict[str, list[Path]] = {c: [] for c in classes}
    temp_crop_dir = DEST_DIR / "_temp"
    temp_crop_dir.mkdir(parents=True, exist_ok=True)

    # 1. Process Custom Dataset (LabelMe Annotations)
    if CUSTOM_SOURCE_DIR.exists():
        for folder_name, class_name in FOLDER_CLASS_MAP.items():
            folder = CUSTOM_SOURCE_DIR / folder_name
            if folder.exists():
                imgs = collect_and_crop_custom(folder, class_name, temp_crop_dir)
                class_images[class_name].extend(imgs)

    # 1b. Process Custom Dataset (Raw Image folders)
    if CUSTOM_SOURCE_DIR.exists():
        for folder_name, class_name in IMAGE_FOLDER_MAP.items():
            folder = CUSTOM_SOURCE_DIR / folder_name
            if folder.exists():
                imgs = collect_raw_images(folder, class_name, temp_crop_dir)
                class_images[class_name].extend(imgs)

    # 2. Process Kaggle Dataset
    if KAGGLE_SOURCE_DIR.exists():
        for folder_name, class_name in KAGGLE_MAP.items():
            imgs = collect_kaggle_images(KAGGLE_SOURCE_DIR, folder_name, temp_crop_dir)
            class_images[class_name].extend(imgs)

    # 3. Split and move
    copied = {"train": 0, "validation": 0, "test": 0}
    
    for cls, imgs in class_images.items():
        if not imgs:
            continue
        
        random.shuffle(imgs)
        n = len(imgs)
        n_train = int(n * SPLIT["train"])
        n_val   = int(n * SPLIT["validation"])

        splits = {
            "train": imgs[:n_train],
            "validation": imgs[n_train:n_train + n_val],
            "test":  imgs[n_train + n_val:],
        }

        for split_name, split_imgs in splits.items():
            dest_cls_dir = DEST_DIR / split_name / cls
            for i, img_path in enumerate(split_imgs):
                dest_name = f"{cls.lower()}_{split_name}_{i:05d}.jpg"
                dest_file = dest_cls_dir / dest_name
                try:
                    shutil.move(str(img_path), str(dest_file))
                    copied[split_name] += 1
                except:
                    pass

    if temp_crop_dir.exists():
        shutil.rmtree(temp_crop_dir)

    print(f"\n--- Dataset Split Results ---")
    for split in ["train", "validation", "test"]:
        for cls in classes:
            d = DEST_DIR / split / cls
            count = len(list(d.glob("*.jpg")))
            print(f"  {split}/{cls}: {count} files")

    print(f"\n[SUCCESS] Master Dataset ready at: {DEST_DIR.resolve()}")

if __name__ == "__main__":
    main()
