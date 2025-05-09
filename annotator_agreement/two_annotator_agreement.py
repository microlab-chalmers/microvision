# Annotator agreement calculator

## PROBLEM STATEMENT ##
# We calculate the annotator agreement for the object labelling task in two steps
# Step 1: Calculate the agreement between the pair of annotators for each of the classes i.e. the percentage of objects that are labelled the same by both annotators
# Step 2: Calculate the average IoU for each of the classes that are labelled the same by both annotators

## CODE ##
import os
import numpy as np
import json
from scipy.optimize import linear_sum_assignment

# Directories
image_dir = "./images"
label_dirs = ["./labels/annotator1", "./labels/annotator2"]

# IoU threshold for matching boxes
iou_threshold = 0.5

# Original image dimensions
image_width = 1920
image_height = 1080

# Pre-defined class names
class_names = {
    0: "person",
    1: "bicycle",
    2: "cyclist",
    3: "e-scooter",
    4: "e-scooterist"
}

def load_annotations(label_dirs):
    """
    Load annotations from multiple annotators and organize them by image and class.
    """
    annotations = {}
    for annotator_id, label_dir in enumerate(label_dirs):
        # Iterate over all files in the annotator's directory
        for filename in os.listdir(label_dir):
            if filename.endswith('.txt'):
                # Construct the full path to the label file
                label_file = os.path.join(label_dir, filename)
                # Extract image ID from the filename
                image_id = os.path.splitext(filename)[0]
                if image_id not in annotations:
                    annotations[image_id] = {}
                with open(label_file, "r") as f:
                    lines = f.readlines()
                    for line in lines:
                        parts = line.strip().split()
                        cls = int(parts[0])
                        bbox = [float(x) for x in parts[1:5]]
                        # Convert bbox to absolute coordinates
                        bbox[0], bbox[2] = bbox[0] * image_width, bbox[2] * image_width
                        bbox[1], bbox[3] = bbox[1] * image_height, bbox[3] * image_height
                        if cls not in annotations[image_id]:
                            annotations[image_id][cls] = {}
                        if annotator_id not in annotations[image_id][cls]:
                            annotations[image_id][cls][annotator_id] = []
                        annotations[image_id][cls][annotator_id].append(bbox)
    return annotations


def bbox_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes in YOLO format.
    """
    # Compute IoU between two YOLO format boxes
    x1_min = box1[0] - box1[2] / 2
    x1_max = box1[0] + box1[2] / 2
    y1_min = box1[1] - box1[3] / 2
    y1_max = box1[1] + box1[3] / 2

    x2_min = box2[0] - box2[2] / 2
    x2_max = box2[0] + box2[2] / 2
    y2_min = box2[1] - box2[3] / 2
    y2_max = box2[1] + box2[3] / 2

    xi1 = max(x1_min, x2_min)
    yi1 = max(y1_min, y2_min)
    xi2 = min(x1_max, x2_max)
    yi2 = min(y1_max, y2_max)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)

    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area != 0 else 0

def calculate_agreement(annotations):
    """
    Calculate agreement and average IoU between annotators for each class.
    """
    class_agreement = {}
    class_iou = {}
    # Iterate over annotations for each image
    for image_id in annotations:
        # We are only interested in classes that are labelled by both annotators
        # Iterate over classes within the image
        for cls in annotations[image_id]:
            # Load the labels
            annotators = annotations[image_id][cls]
            # Check if all annotators have labeled this class in the image
            if len(annotators) < len(label_dirs):
                continue  # Skip if any annotator missed this class
            # Collect bounding boxes from all annotators
            boxes_a = annotators[0]  # Bounding boxes from annotator 0
            boxes_b = annotators[1]  # Bounding boxes from annotator 1
            # Compute IoU matrix between all pairs of boxes
            iou_matrix = np.zeros((len(boxes_a), len(boxes_b)))
            for i, box_a in enumerate(boxes_a):
                for j, box_b in enumerate(boxes_b):
                    iou = bbox_iou(box_a, box_b)
                    # We use negative IoU because the linear_sum_assignment function minimizes the cost
                    iou_matrix[i, j] = -iou
            # Solve the assignment problem
            row_ind, col_ind = linear_sum_assignment(iou_matrix)
            matches = 0
            iou_sum = 0
            for i, j in zip(row_ind, col_ind):
                iou = -iou_matrix[i, j]  # Convert back to positive IoU
                if iou >= iou_threshold:
                    matches += 1
                    iou_sum += iou
            # Total number of boxes is the maximum between both annotators
            total_boxes = max(len(boxes_a), len(boxes_b))
            if cls not in class_agreement:
                class_agreement[cls] = []
                class_iou[cls] = []
            # Calculate agreement and average IoU for the class
            agreement = matches / total_boxes if total_boxes > 0 else 0
            avg_iou_per_class = iou_sum / matches if matches > 0 else 0
            class_agreement[cls].append(agreement)
            class_iou[cls].append(avg_iou_per_class)
    return class_agreement, class_iou

annotations = load_annotations(label_dirs)
class_agreement, class_iou = calculate_agreement(annotations)

# Print results
overall_agreement = []
overall_iou = []
for cls in class_agreement:
    avg_agreement = np.mean(class_agreement[cls])
    avg_iou = np.mean(class_iou[cls])
    overall_agreement.extend(class_agreement[cls])
    overall_iou.extend(class_iou[cls])
    print(f"Class {class_names[cls]}: Agreement={avg_agreement*100:.2f}%, Average IoU={avg_iou:.4f}")

print(f"Overall Agreement: {np.mean(overall_agreement)*100:.2f}%")
print(f"Overall Average IoU: {np.mean(overall_iou):.4f}")
