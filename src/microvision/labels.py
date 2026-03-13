import json
from pathlib import Path

import numpy as np
from supervision import Detections
from tqdm import tqdm

from microvision.config import IMG_HEIGHT, IMG_WIDTH, OBJECT_CLASSES


def get_boxes(path_labels: str):
    """Get bounding boxes and classes from label file (YOLO format)

    Args:
        path_labels (str): Path to txt label file (in YOLO format)

    Returns:
        (x0, y0, x1, y1) (list): Lists of box coordinates
        class_ids (list): List of class IDs
    """
    x0, y0, x1, y1 = [], [], [], []
    class_ids = []

    with open(path_labels, "r") as file:
        for line in file:
            _split = line.split()

            class_number = int(_split[0])

            _x_center, _y_center = (
                float(_split[1]) * IMG_WIDTH,
                float(_split[2]) * IMG_HEIGHT,
            )
            _width, _height = (
                float(_split[3]) * IMG_WIDTH,
                float(_split[4]) * IMG_HEIGHT,
            )

            x0.append(_x_center - _width / 2)
            y0.append(_y_center - _height / 2)
            x1.append(_x_center + _width / 2)
            y1.append(_y_center + _height / 2)

            class_ids.append(class_number)

    return (x0, y0, x1, y1), class_ids


def labels2coco(yolo_annotation, image_id, img_width, img_height, anno_id_start):
    """Convert individual labels file from YOLO to COCO format

    Args:
        yolo_annotation (_type_): _description_
        image_id (_type_): _description_
        img_width (_type_): _description_
        img_height (_type_): _description_
        anno_id_start (_type_): _description_

    Returns:
        _type_: _description_
    """
    annotations = []
    for anno_line in yolo_annotation.splitlines():
        parts = anno_line.strip().split()

        # Convert object class (needs to start from 1)
        category_id = int(parts[0]) + 1

        bbox = [float(x) for x in parts[1:5]]

        # Convert box to COCO format
        x_center, y_center, bbox_width, bbox_height = bbox
        x_top_left = (x_center - bbox_width / 2) * img_width
        y_top_left = (y_center - bbox_height / 2) * img_height
        width = bbox_width * img_width
        height = bbox_height * img_height

        annotations.append(
            {
                "id": anno_id_start,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [x_top_left, y_top_left, width, height],
                "area": width * height,
                "iscrowd": 0,
            }
        )

        anno_id_start += 1

    return annotations, anno_id_start


def yolo2coco(
    path_yolo: str,
    path_json: str,
    obj_classes: dict[int, str] = OBJECT_CLASSES,
    img_width: int = IMG_WIDTH,
    img_height: int = IMG_HEIGHT,
):
    """Convert labels from YOLO to COCO JSON

    Args:
        path_yolo (str): Path to YOLO labels
        path_json (str): Path to output COCO JSON
        obj_classes (dict, optional): Object classes. Defaults to OBJECT_CLASSES.
        img_width (int, optional): Image width. Defaults to IMG_WIDTH.
        img_height (int, optional): Image height. Defaults to IMG_HEIGHT.
    """
    # Define label categories
    coco_dataset = {
        "images": [],
        "annotations": [],
        "categories": [{"id": k + 1, "name": v} for k, v in obj_classes.items()],
    }

    annotation_id = 1

    images = list(Path(path_yolo, "images").glob("*"))

    for image in tqdm(images):
        image_id = image.stem

        labels = Path(path_yolo, "labels", image_id + ".txt")

        # Convert labels if exist (if not, the image is a "background image" without any objects)
        if labels.exists():
            with open(labels, "r") as file:
                yolo_annotation = file.read()

            coco_anno, annotation_id = labels2coco(
                yolo_annotation, image_id, img_width, img_height, annotation_id
            )

            coco_dataset["annotations"].extend(coco_anno)

        coco_dataset["images"].append(
            {
                "id": image_id,
                "width": img_width,
                "height": img_height,
                "file_name": f"{image.name}",
            }
        )

    # Save as json file
    with open(path_json, "w") as outfile:
        outfile.write(json.dumps(coco_dataset))


# YOLO2Roboflow


def anno2detections(path_labels):

    (x0, y0, x1, y1), class_ids = get_boxes(path_labels)

    boxes = np.array([x0, y0, x1, y1]).T

    class_ids = np.array(class_ids)

    # Detections
    detections = Detections(
        xyxy=boxes,
        class_id=class_ids,
    )

    # Labels
    labels = [f"{OBJECT_CLASSES[_class_id]}" for _class_id in class_ids]

    return detections, labels
