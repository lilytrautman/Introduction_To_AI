import os
import cv2
import numpy as np
from feature_extraction import extract_features, TextVectorizer

RESULTS_DIR = 'results'  # Folder containing cropped images

# Helper: Recursively collect all image file paths from a directory
def collect_image_paths(folder, exts={'.jpg', '.jpeg', '.png', '.bmp'}):
    image_paths = []
    for root, _, files in os.walk(folder):
        for file in files:
            if os.path.splitext(file)[1].lower() in exts:
                image_paths.append(os.path.join(root, file))
    return image_paths

# Main: Build feature matrix and label vector
def build_feature_matrix(image_folder, label_map=None, text_vectorizer=None):
    image_paths = collect_image_paths(image_folder)
    features = []
    texts = []
    labels = []
    # First pass: extract text for all images (for fitting vectorizer)
    for img_path in image_paths:
        img = cv2.imread(img_path)
        _, text = extract_features(img)
        texts.append(text)
    # Fit text vectorizer if not provided
    if text_vectorizer is None:
        text_vectorizer = TextVectorizer()
        text_vectorizer.fit(texts)
    # Second pass: extract all features
    for img_path in image_paths:
        img = cv2.imread(img_path)
        feats, text = extract_features(img, text_vectorizer)
        # Label extraction: expects label_map or folder name as label
        if label_map:
            label = label_map.get(os.path.basename(img_path), None)
        else:
            label = os.path.basename(os.path.dirname(img_path))
        # Only add if label is not None or empty
        if label is not None and label != "":
            features.append(feats)
            labels.append(label)
        # else: skip image with missing label
    X = np.array(features)
    y = np.array(labels)
    return X, y, text_vectorizer

if __name__ == "__main__":
    X, y, text_vectorizer = build_feature_matrix(RESULTS_DIR)
    print(f"Feature matrix shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
