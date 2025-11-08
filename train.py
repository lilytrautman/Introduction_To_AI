# train.py
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from sklearn.model_selection import train_test_split
from dataset import CargoDataset

def collate_fn(batch):
    # Filter out None entries (skipped images)
    batch = [b for b in batch if b is not None]
    if not batch:
        return [], []
    return tuple(zip(*batch))


# Use torchvision's Faster R-CNN with ResNet-50 FPN backbone, pretrained on COCO
def get_fasterrcnn_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=True)
    # Replace the head for dataset (SKU10000: 1 class + background)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    # Increase the max detections per image (default is 100)
    model.roi_heads.detections_per_img = 500
    return model

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    num_classes = 2  # 1 class ('object') + background

    # Update these paths to your SKU10000 dataset
    train_dir = 'SKU10000-1/train'
    train_ann_file = 'SKU10000-1/train/_annotations.coco.json'
    transform = T.Compose([T.ToTensor()])
    
    dataset = CargoDataset(root=train_dir, annFile=train_ann_file, transforms=transform)
    
    print("Splitting dataset into training and validation sets...")
    indices = list(range(len(dataset)))
    train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42)

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=4, collate_fn=collate_fn)

    print("Initializing model...")
    model = get_fasterrcnn_model(num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    
    num_epochs = 50
    EARLY_STOPPING_PATIENCE = 5
    best_val_loss = float('inf')
    epochs_no_improve = 0

    print("Starting training...")

    for epoch in range(num_epochs):
        # --- TRAINING LOOP ---
        model.train()
        train_loss = 0
        for images, targets in train_loader:
            # Skip empty batches
            if not images or not targets:
                continue
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            train_loss += losses.item()
        
        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}")

        # --- VALIDATION LOOP ---
        val_loss = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                
                # Temporarily set model to training mode to get the loss dictionary
                model.train()
                loss_dict = model(images, targets)
                model.eval() # Set back to evaluation mode
                
                losses = sum(loss for loss in loss_dict.values())
                val_loss += losses.item()

        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}")

        # --- EARLY STOPPING ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'cargo_detector.pth')
            print("Validation loss improved. Model saved.")
        else:
            epochs_no_improve += 1
            print(f"No improvement in validation loss for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered. Training stopped.")
            break
            
    print("Training finished.")
    # Print total skipped images
    if hasattr(dataset, 'skipped_images'):
        print(f"Total images skipped due to missing files: {dataset.skipped_images}")

if __name__ == "__main__":
    main()