import os
import shutil
import csv
import glob
import subprocess

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, 'Packaged-Grocery-store-items')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
DB_DIR = os.path.join(BASE_DIR, 'Pkg-Items-PreLabeled')
CSV_PATH = os.path.join(DB_DIR, 'prelabeled_items.csv')
PREDICT_SCRIPT = os.path.join(BASE_DIR, 'predict.py')

# Ensure DB directory exists
os.makedirs(DB_DIR, exist_ok=True)

# Prepare CSV file
csv_file = open(CSV_PATH, 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['image_filename', 'category'])

# Dynamically find category names from subdirectories in SRC_DIR
CATEGORY_NAMES = [d for d in os.listdir(SRC_DIR) if os.path.isdir(os.path.join(SRC_DIR, d))]
print(f"Found categories: {CATEGORY_NAMES}")

# Process each category folder
for category in CATEGORY_NAMES:
    category_folder = os.path.join(SRC_DIR, category)
    if not os.path.isdir(category_folder):
        print(f"Warning: {category_folder} does not exist.")
        continue
    image_files = glob.glob(os.path.join(category_folder, '*.*'))
    for img_idx, img_path in enumerate(image_files):
        # Clean results folder before running predict.py
        for f in glob.glob(os.path.join(RESULTS_DIR, '*.*')):
            os.remove(f)
        # Run predict.py on image
        subprocess.run(['python', PREDICT_SCRIPT, img_path], check=True)
        # Find new images in results folder (cropped items)
        result_images = glob.glob(os.path.join(RESULTS_DIR, '*.*'))
        for res_idx, res_img in enumerate(result_images):
            # Use unique name for each cropped item
            new_name = f"{category}_{img_idx}_{res_idx}{os.path.splitext(res_img)[1]}"
            new_path = os.path.join(DB_DIR, new_name)
            shutil.move(res_img, new_path)
            # Write to CSV
            csv_writer.writerow([new_name, category])

csv_file.close()
print(f"Pre-labeled dataset created at {DB_DIR} with CSV: {CSV_PATH}")
