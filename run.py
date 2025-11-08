
# Main script to run prediction and classification on a single image using trained Random Forest
import subprocess
import sys
import os
import cv2
import shutil
import json
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import joblib
from feature_extraction import extract_features

TEXT_VECTORIZER_PATH = 'text_vectorizer.pkl'
LABEL_MAP_PATH = 'idx_to_label.pkl'

DB_FOLDER = 'results'  # Folder where cropped images are saved after detection
MODEL_PATH = 'random_forest_model.pkl'

def run_classification(image_path):
    # Clean the results directory before running a new prediction
    if os.path.exists(DB_FOLDER):
        shutil.rmtree(DB_FOLDER)
    os.makedirs(DB_FOLDER, exist_ok=True)

    # 1. Run predict.py on image_path (detect items)
    print(f"Running detection on {image_path}...")
    subprocess.run([sys.executable, "predict.py", image_path], check=True)

    # 2. Load detection data from JSON
    detections_path = os.path.join(DB_FOLDER, 'detections.json')
    if not os.path.exists(detections_path):
        print("No detected item images found in results folder.")
        return
    with open(detections_path, 'r') as f:
        detections = json.load(f)

    # 3. Load trained model, text vectorizer, and label map
    clf = joblib.load(MODEL_PATH)
    text_vectorizer = joblib.load(TEXT_VECTORIZER_PATH)
    idx_to_label = joblib.load(LABEL_MAP_PATH)

    # 4. Classify each cropped image and store the result
    print("Classifying detected items...")
    for detection in detections:
        img_name = detection['filename']
        img_path = os.path.join(DB_FOLDER, img_name)
        image = cv2.imread(img_path)
        if image is None:
            detection['category'] = 'Unknown'
            continue
        feat, _ = extract_features(image, text_vectorizer)
        feat = np.array(feat).reshape(1, -1)
        pred_idx = clf.predict(feat)[0]
        detection['category'] = idx_to_label.get(pred_idx, "Unknown")

    # 5. Sort cropped images into category folders (old behavior)
    print("Organizing cropped images into category folders...")
    predicted_folder = os.path.join(DB_FOLDER, "predicted")
    os.makedirs(predicted_folder, exist_ok=True)
    for detection in detections:
        category_name = detection['category']
        img_name = detection['filename']
        src_path = os.path.join(DB_FOLDER, img_name)
        if not os.path.exists(src_path): continue

        category_folder = os.path.join(predicted_folder, category_name)
        os.makedirs(category_folder, exist_ok=True)
        dest_path = os.path.join(category_folder, img_name)
        shutil.copy(src_path, dest_path) # Use copy to keep original for final image
    
    # 6. Draw final annotated image
    print("Generating final annotated image...")
    original_image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(original_image)
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()

    for detection in detections:
        draw.rectangle(detection['box'], outline="lime", width=3)
        text_position = (detection['box'][0], detection['box'][1] - 20)
        draw.text(text_position, detection['category'], fill="lime", font=font)

    final_output_path = os.path.join(DB_FOLDER, "final_classified_output.jpg")
    original_image.save(final_output_path)
    print(f"\nComplete! Check the results folder for the final annotated image:\n{final_output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run.py <image_path>")
        sys.exit(1)
    image_path = sys.argv[1]
    run_classification(image_path)
