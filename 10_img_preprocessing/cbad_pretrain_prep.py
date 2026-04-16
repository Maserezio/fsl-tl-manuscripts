import os
import cv2
import glob
from pathlib import Path
from tqdm.notebook import tqdm

INPUT_DATASET_DIR = "../../../READ-ICDAR2019-cBAD-dataset-blind/" 

OUTPUT_DIR = "../../../cbad_prepr"
TILE_SIZE = 512
STRIDE = 384  # Overlap helps preserve context at edges

def tile_image(img_path, save_dir):
    img = cv2.imread(str(img_path))
    if img is None: return 0
    
    h, w, _ = img.shape
    base_name = img_path.stem
    count = 0
    
    # Sliding window
    for y in range(0, h, STRIDE):
        for x in range(0, w, STRIDE):
            # Check if tile goes out of bounds
            y_end = min(h, y + TILE_SIZE)
            x_end = min(w, x + TILE_SIZE)
            
            # Correct crop coordinates if at the edge
            y_start = y_end - TILE_SIZE if y_end == h else y
            x_start = x_end - TILE_SIZE if x_end == w else x
            
            # Ensure indices are valid (image might be smaller than tile)
            if y_start < 0: y_start = 0
            if x_start < 0: x_start = 0
            
            crop = img[y_start:y_end, x_start:x_end]
            
            # Skip if crop is empty or too small (e.g. edge cases)
            if crop.shape[0] < 224 or crop.shape[1] < 224:
                continue

            save_path = os.path.join(save_dir, f"{base_name}_{y}_{x}.jpg")
            cv2.imwrite(save_path, crop)
            count += 1
    return count

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
image_extensions = ['*.jpg', '*.png', '*.bmp', '*.tif']
all_files = []

for ext in image_extensions:
    all_files.extend(Path(INPUT_DATASET_DIR).rglob(ext))

print(f"Found {len(all_files)} source images.")

total_tiles = 0
for file in tqdm(all_files):
    total_tiles += tile_image(file, OUTPUT_DIR)

print(f"✅ Processing complete.")
print(f"Generated {total_tiles} tiles in '{OUTPUT_DIR}'.")
