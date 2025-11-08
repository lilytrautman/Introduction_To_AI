# dataset.py
import torch
from torchvision.datasets import CocoDetection
from PIL import Image

class CargoDataset(CocoDetection):
    def __init__(self, root, annFile, transforms=None):
        super(CargoDataset, self).__init__(root, annFile)
        self.transforms = transforms
        self.skipped_images = 0

    def __getitem__(self, idx):
        # We override the original __getitem__ to fix the transform issue.
        
        # 1. Load image and annotations using the coco API
        coco = self.coco
        img_id = self.ids[idx]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)
        
        path = coco.loadImgs(img_id)[0]['file_name']
        try:
            img = Image.open(f"{self.root}/{path}").convert('RGB')
        except FileNotFoundError:
            self.skipped_images += 1
            return None

        # 2. Format the target dictionary and filter invalid boxes
        boxes = []
        labels = []
        for t in target:
            xmin, ymin, width, height = t['bbox']
            
            # Skip boxes with zero area.
            if width <= 0 or height <= 0:
                continue
            
            boxes.append([xmin, ymin, xmin + width, ymin + height])
            labels.append(t['category_id'])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        # For SKU10000, all objects are the same class ('object'), so set label to 1 for all
        labels = torch.ones(len(labels), dtype=torch.int64)
        
        final_target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([img_id])
        }

        # Skip entries with no valid bounding boxes
        if boxes.size(0) == 0:
            #print(f"No valid boxes for image ID {img_id}. Skipping this entry.")
            #print(f"Annotations for image ID {img_id}: {target}")  # Log annotations for debugging
            #print(f"Image path: {self.root}/{path}")  # Log image path for debugging
            return self.__getitem__((idx + 1) % len(self.ids))

        # Log the number of valid boxes for debugging
        #print(f"Image ID {img_id} has {boxes.size(0)} valid boxes.")

        # 3. Apply transforms ONLY to the image
        if self.transforms is not None:
            img = self.transforms(img)
        
        return img, final_target