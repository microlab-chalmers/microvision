# Annotator agreement calculator

## PROBLEM STATEMENT ##
# We calculate the annotator agreement for the object labelling task in two steps
# Step 1: Calculate the agreement between the annotators for each of the classes i.e. the percentage of objects that are labelled the same by all annotators
# Step 2: Calculate the average IoU for each of the classes that are labelled the same by all annotators

## CODE ##
import os
import numpy as np
import json
from scipy.optimize import linear_sum_assignment

# Directories
label_dirs = ["./labels/annotator1", "./labels/annotator2", "./labels/annotator3"]

# IoU threshold for matching boxes
iou_threshold = 0.5

# Original image dimensions
image_width = 1920
image_height = 1080

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
    Calculate the agreement among the annotators.
    """
    class_agreement = {}
    class_iou = {}
    for image_id in annotations:
        for cls in annotations[image_id]:
            # Get bounding boxes for all annotators
            annotators_boxes = [annotations[image_id][cls].get(a, []) for a in range(len(label_dirs))]
            # If every annotator has at least one box, compute matches
            if any(len(b) == 0 for b in annotators_boxes):
                continue
            
            ref_boxes = annotators_boxes[0]
            total_boxes = max(len(b) for b in annotators_boxes)
            matches = 0
            iou_sum = 0
            for ref_box in ref_boxes:
                # Check IoU with each annotator's best matching box
                ious = []
                for other_boxes in annotators_boxes[1:]:
                    best_iou = max(bbox_iou(ref_box, ob) for ob in other_boxes) if other_boxes else 0
                    ious.append(best_iou)
                # Only count as match if all IoUs exceed threshold
                if all(i >= iou_threshold for i in ious):
                    matches += 1
                    iou_sum += sum(ious) / len(ious)  # average IoU across all annotators

            # Store metrics
            if cls not in class_agreement:
                class_agreement[cls] = []
                class_iou[cls] = []
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
