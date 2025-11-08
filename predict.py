# predict.py
import torch
import torchvision.transforms as T
import json
from PIL import Image, ImageDraw
from train import get_fasterrcnn_model

# packages necessary to run this code: pip install torch torchvision pillow numpy opencv-python scikit-image scikit-learn pytesseract
# and download this too: https://github.com/tesseract-ocr/tessdoc/blob/main/Installation.md

def main():
    import sys
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)
    img_path = sys.argv[1]
    # Load the model architecture and the trained weights
    model = get_fasterrcnn_model(num_classes = 2)
    model.load_state_dict(torch.load('cargo_detector.pth'))
    model.eval() # Set to evaluation mode
    img = Image.open(img_path).convert("RGB")

    # Prepare the image for the model
    transform = T.Compose([T.ToTensor()])
    img_tensor = transform(img)

    # Make a prediction
    with torch.no_grad():
        prediction = model([img_tensor])

    # Debugging: Print the prediction output
    print(prediction)

    # Draw the results on the image and crop/save each detected item
    draw = ImageDraw.Draw(img)
    confidence_threshold = 0.4 # Adjust this threshold as needed

    detections = 0  # Counter for detections
    cropped_results = []
    detection_data = [] # To store box, score, and filename
    for idx, (box, label, score) in enumerate(zip(prediction[0]['boxes'], prediction[0]['labels'], prediction[0]['scores'])):
        if score > confidence_threshold:
            detections += 1
            box = box.cpu().tolist() # Move tensor to CPU for list conversion
            label_name = "cargo" if label.item() == 1 else "other"
            draw.rectangle(box, outline="red", width=3)
            draw.text((box[0], box[1] - 10), f"{label_name} {score:.2f}", fill="red")

            # Smart resize: expand bounding box by 10%
            x_min, y_min, x_max, y_max = box
            box_width = x_max - x_min
            box_height = y_max - y_min
            margin_x = box_width * 0.1
            margin_y = box_height * 0.1

            # Calculate expanded box coordinates
            img_width, img_height = img.size
            new_x_min = max(0, x_min - margin_x)
            new_y_min = max(0, y_min - margin_y)
            new_x_max = min(img_width, x_max + margin_x)
            new_y_max = min(img_height, y_max + margin_y)

            # Crop and save the detected item with smart resize
            cropped_img = img.crop((new_x_min, new_y_min, new_x_max, new_y_max))
            # Preserve aspect ratio: resize so the longer side is 224, shorter side scales proportionally
            orig_w, orig_h = cropped_img.size
            if orig_w > orig_h:
                new_w = 224
                new_h = int(orig_h * (224 / orig_w))
            else:
                new_h = 224
                new_w = int(orig_w * (224 / orig_h))
            cropped_img = cropped_img.resize((new_w, new_h), Image.LANCZOS)
            result_filename = f"detected_item_{idx+1}.jpg"
            result_path = f"results/{result_filename}"
            cropped_img.save(result_path)
            cropped_results.append(result_path)
            detection_data.append({
                "box": box,
                "score": score.item(),
                "filename": result_filename
            })

    # Log the number of detections
    if detections == 0:
        print("No objects detected above the confidence threshold.")
    else:
        print(f"Number of detections: {detections}")
        print("Cropped detected items saved:")
        for path in cropped_results:
            print(f"- {path}")

    # Save detection data to a JSON file for run.py to use
    with open("results/detections.json", "w") as f:
        json.dump(detection_data, f)
    print("Detection data saved to results/detections.json")

    # Save the annotated image in results folder
    annotated_path = "results/prediction_result.jpg"
    img.save(annotated_path)
    print(f"Prediction result saved as '{annotated_path}'")

    # img.show()

if __name__ == "__main__":
    main()