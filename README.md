# Introduction_To_AI

## How to Run This Project

There are many files in this project. Here’s how to actually run it:

### Required Packages

Install the following Python packages:

```bash
pip install torch torchvision pillow numpy opencv-python scikit-image scikit-learn pytesseract joblib
```

**Also required:**  
Tesseract OCR engine ([installation guide](https://github.com/tesseract-ocr/tessdoc/blob/main/Installation.md))

---

### Steps to Run

1. **Download SKU100000-1 dataset**  
    Run:
    ```bash
    python download_data.py
    ```

2. **Train a model using Faster R-CNN**  
    Run:
    ```bash
    python train.py
    ```
    > *Note: Training takes about 3 hours on an 8GB Quadro RTX 4000 GPU.*

3. **Create the dataset for Random Forest**  
    Run:
    ```bash
    python automated_labeling.py
    ```
    This pulls images from the `Packaged-Grocery-store-items` folder and uses Faster R-CNN object detection to break them down into individual items in (hopefully) the correct categories.
    Make sure to rename the resulting directory to Pkg-Items-PreLabeled-1 (this is to prevent accidently overwriting).
    NOTE MISSING PHOTOS FOR PRODUCE

4. **Train Random Forest**  
    Run:
    ```bash
    python train_forest.py
    ```
    > *Note: Training takes about 25 minutes.*

    ## My Results

    **Category Counts:**  
    `{'Grains': 665, 'Dairy': 624, 'Meats': 596, 'Fruit': 425}`

    **Test Accuracy:**  
    `0.8182`

    **Confusion Matrix:**
    ```
    [[113   6   9   8]
     [ 14  28  13  13]
     [  2   3 124   0]
     [ 10   2   4 113]]
    ```

    **Classification Report:**
    ```
                  precision    recall  f1-score   support

           Dairy       0.81      0.83      0.82       136
           Fruit       0.72      0.41      0.52        68
          Grains       0.83      0.96      0.89       129
           Meats       0.84      0.88      0.86       129

        accuracy                           0.82       462
       macro avg       0.80      0.77      0.77       462
    weighted avg       0.81      0.82      0.81       462
    ```

5. **Ready to Go!**  
    Run:
    ```bash
    python run.py <image_name>
    ```
    This outputs the detected objects in your image to folders in `results/predicted/<category>/<location>`.


## Example Output
![final_classified_output](https://github.com/user-attachments/assets/d7a75039-6207-4d21-b324-9318e2bd1b21)
