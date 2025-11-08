# download_data.py
from roboflow import Roboflow


# Paste your private API key here
rf = Roboflow(api_key="5U6WPpUcvUoNJ528z5cL")
# Download SKU10000 dataset from Roboflow
project = rf.workspace("dataconversion").project("sku10000")
# Use the latest version available (update if needed)
dataset = project.version(1).download("coco")

print(f"Dataset downloaded to: {dataset.location}")