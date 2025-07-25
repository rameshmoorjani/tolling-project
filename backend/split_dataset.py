import os
import shutil
import random
from collections import defaultdict

# Set paths
CROPS_DIR = 'data/processed/crops'
OUTPUT_DIR = 'data/processed/classification'
TRAIN_RATIO = 0.8
CLASSES = ['Car', 'Truck', 'Cyclist']

# Create target train/val folders
for split in ['train', 'val']:
    for cls in CLASSES:
        split_path = os.path.join(OUTPUT_DIR, split, cls)
        os.makedirs(split_path, exist_ok=True)

# Group images by class
images_by_class = defaultdict(list)
for fname in os.listdir(CROPS_DIR):
    for cls in CLASSES:
        if f"_{cls}.png" in fname:
            images_by_class[cls].append(fname)

# Shuffle and split each class
for cls, files in images_by_class.items():
    random.shuffle(files)
    train_count = int(len(files) * TRAIN_RATIO)

    for i, fname in enumerate(files):
        split = 'train' if i < train_count else 'val'
        src = os.path.join(CROPS_DIR, fname)
        dst = os.path.join(OUTPUT_DIR, split, cls, fname)
        shutil.copy2(src, dst)

print("✅ Dataset split complete.")
