import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from tqdm import tqdm  # For progress bar
from sklearn.metrics import average_precision_score


def load_model(model_path):
    #model = torch.load(model_path)
    #model.eval()  # Set model to evaluation mode
    checkpoint = torch.load(model_path)

    # Extract the model from the checkpoint dictionary
    model = checkpoint['model'] # Modify the key based on your checkpoint structure

# Set the model to evaluation mode
    model.eval()
    return model


def load_validation_data(validation_data_path):
    # Load validation image paths and labels
    # Modify this function according to your dataset structure
    validation_images = [...]  # List of image file paths
    validation_labels = [...]  # Corresponding labels for each image
    return validation_images, validation_labels


def preprocess_image(image_path):
    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize((416, 416)),
        transforms.ToTensor(),
    ])
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor


def calculate_metrics(gt_boxes, pred_boxes, iou_threshold=0.5):
    # Calculate average precision or other metrics
    # You can use libraries like scikit-learn or custom functions
    # based on your evaluation criteria
    average_precision = average_precision_score(gt_boxes, pred_boxes)
    return average_precision


def validate_yolov8(model, validation_images, validation_labels):
    model.eval()  # Set model to evaluation mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    metrics = []

    for image_path, gt_boxes in tqdm(zip(validation_images, validation_labels), total=len(validation_images),
                                     desc="Validation"):
        image_tensor = preprocess_image(image_path)
        image_tensor = image_tensor.to(device)

        with torch.no_grad():
            # Perform inference
            outputs = model(image_tensor)
            # Process model outputs to get predicted boxes

        # Calculate metrics based on ground truth and predicted boxes
        # Here, you should convert box format and compute IoU for example

        metric = calculate_metrics(gt_boxes, pred_boxes)
        metrics.append(metric)

    avg_metric = np.mean(metrics)
    return avg_metric


if __name__ == "__main__":
    model_path = "yolov8m.pt"
    validation_data_path = "/home/kenaro/ForestFireDetection/AI-Yolo/Wildfire-2/valid/"

    model = load_model(model_path)
    validation_images, validation_labels = load_validation_data(validation_data_path)

    avg_metric = validate_yolov8(model, validation_images, validation_labels)
    print(f"Average Validation Metric: {avg_metric}")
