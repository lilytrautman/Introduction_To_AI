import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.feature_extraction.text import TfidfVectorizer
import pytesseract
import platform

# Set Tesseract path based on OS
if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"
elif platform.system() == "Darwin":  # macOS
    pytesseract.pytesseract.tesseract_cmd = "/usr/local/bin/tesseract"
elif platform.system() == "Linux":
    pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

# HSV Color Histogram

def extract_hsv_histogram(image, bins=(8, 8, 8)):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

# Local Binary Pattern (LBP)
def extract_lbp_features(image, P=8, R=1.0, method='uniform'):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P, R, method)
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), range=(0, P + 2))
    hist = hist.astype('float')
    hist /= (hist.sum() + 1e-6)
    return hist

# OCR Text Extraction
def extract_text(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)
    return text

# Text Vectorization (fit on all texts, then transform)
class TextVectorizer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=100)
    def fit(self, texts):
        self.vectorizer.fit(texts)
    def transform(self, texts):
        return self.vectorizer.transform(texts).toarray()

# Full feature extraction pipeline for one image
def extract_features(image, text_vectorizer=None):
    hsv_hist = extract_hsv_histogram(image)
    lbp_hist = extract_lbp_features(image)
    text = extract_text(image)
    text_features = None
    if text_vectorizer is not None:
        text_features = text_vectorizer.transform([text])[0]
    else:
        text_features = np.zeros(100)  # Placeholder if not fitted
    features = np.concatenate([hsv_hist, lbp_hist, text_features])
    return features, text
