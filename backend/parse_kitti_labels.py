import os
import cv2
from tqdm import tqdm

# Define paths relative to project root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LABEL_DIR = os.path.join(BASE_DIR, 'data/raw/label_2/training/label_2')
IMAGE_DIR = os.path.join(BASE_DIR, 'data/raw/image_2/training/image_2')
OUTPUT_DIR = os.path.join(BASE_DIR, 'data/processed/crops')

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Only keep these object types
VALID_CLASSES = {"Car", "Truck", "Cyclist"}

# Loop through all label files
for label_file in tqdm(sorted(os.listdir(LABEL_DIR)), desc="Processing labels"):
    if not label_file.endswith(".txt"):
        continue

    base_filename = os.path.splitext(label_file)[0]
    image_path = os.path.join(IMAGE_DIR, base_filename + ".png")
    image = cv2.imread(image_path)

    if image is None:
        print(f"⚠️  Image not found or unreadable: {image_path}")
        continue

    label_path = os.path.join(LABEL_DIR, label_file)
    with open(label_path, 'r') as f:
        lines = f.readlines()

    for idx, line in enumerate(lines):
        parts = line.strip().split()
        obj_class = parts[0]

        if obj_class not in VALID_CLASSES:
            continue

        # Extract and convert bbox coordinates
        try:
            xmin, ymin, xmax, ymax = map(int, map(float, parts[4:8]))
        except ValueError:
            print(f"⚠️  Invalid bbox in {label_file}, line {idx}")
            continue

        # Validate bbox range
        if xmin >= xmax or ymin >= ymax:
            continue

        # Crop and save
        crop = image[ymin:ymax, xmin:xmax]
        if crop.size == 0:
            continue

        crop_filename = f"{base_filename}_{idx}_{obj_class}.png"
        crop_path = os.path.join(OUTPUT_DIR, crop_filename)
        cv2.imwrite(crop_path, crop)
