import os
import csv
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
from feature_extraction import extract_features, TextVectorizer
from sklearn.metrics import confusion_matrix, classification_report

DB_FOLDER = 'Pkg-Items-PreLabeled-1'
CSV_PATH = os.path.join(DB_FOLDER, 'prelabeled_items.csv')
MODEL_PATH = 'random_forest_model.pkl'

# Step 1: Load CSV and filter valid images
image_paths = []
labels = []
from collections import Counter
with open(CSV_PATH, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        img_path = os.path.join(DB_FOLDER, row['image_filename'])
        if os.path.isfile(img_path):
            image_paths.append(img_path)
            labels.append(row['category'])
        else:
            print(f"Skipping missing image: {row['image_filename']}")

# Print category counts after loading
print("Category counts in training data:")
print(Counter(labels))

import cv2
# Step 2: Extract features (with text vectorizer)
all_texts = []
valid_image_paths = []
valid_labels = []
for img_path in image_paths:
    image = cv2.imread(img_path)
    if image is None:
        print(f"Warning: Unable to load image {img_path}, skipping.")
        continue
    _, text = extract_features(image)
    all_texts.append(text)
    valid_image_paths.append(img_path)
    valid_labels.append(labels[image_paths.index(img_path)])

# Fit text vectorizer
text_vectorizer = TextVectorizer()
text_vectorizer.fit(all_texts)

features = []
for img_path in valid_image_paths:
    image = cv2.imread(img_path)
    feat, _ = extract_features(image, text_vectorizer)
    features.append(feat)
features = np.array(features)

# Step 3: Encode labels
unique_labels = sorted(set(valid_labels))
label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
idx_to_label = {idx: label for label, idx in label_to_idx.items()} # Create inverse map for prediction
labels_encoded = np.array([label_to_idx[label] for label in valid_labels])

# Step 4: Train/test split
X_train, X_test, y_train, y_test = train_test_split(features, labels_encoded, test_size=0.2, random_state=42)

# Step 5: Train Random Forest
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Step 6: Evaluate
acc = clf.score(X_test, y_test)
print(f"Test accuracy: {acc:.4f}")
y_pred = clf.predict(X_test)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=unique_labels))

# Step 7: Save model and text vectorizer
joblib.dump(clf, MODEL_PATH)
TEXT_VECTORIZER_PATH = 'text_vectorizer.pkl'
joblib.dump(text_vectorizer, TEXT_VECTORIZER_PATH)
joblib.dump(idx_to_label, "idx_to_label.pkl")
print(f"Model saved to {MODEL_PATH}")
print(f"Text vectorizer saved to {TEXT_VECTORIZER_PATH}")
print("Label mapping saved to idx_to_label.pkl")
