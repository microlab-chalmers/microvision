# MicroVision dataset

The MicroVision dataset is provided in three files:

- images.zip: Contains all images
    - JPG format
    - Faces and license plates were blurred using BrighterAI's Precision Blur technology
    - File naming: "S{X}_{Y}.jpg" (where X is the number of the scene, and Y the number of the frame within the scene, both starting from zero)

- labels.zip: Contains all labels in YOLO format (object box coordinates and classes)
    - TXT format
    - Same file naming as for images

- meta.csv: Contains meta data e.g. needed to reproduce the splits used for our benchmark model
    - CSV format
    - Columns:
        - "scene": Scene identifier
        - "img": Image/label name (without extension)
        - "split": Which of the three splits the image is part of (train/val/test) 
        - "split_notest": Which of the two splits the image is part of when training on the full dataset without test set (train/val)

- microvision_yolo11.pt: Weights file for the YOLO11-X model
- microvision_fasterrcnn.pth: Weights file for the Detectron2 Faster R-CNN model
- microvision_rfdetr.pth: Weights file for the RF-DETR large model (resolution 1232 px)