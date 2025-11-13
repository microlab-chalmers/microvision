# Yolo11 Detect and Track
This script uses YOLO11 for object detection and tracking on videos and images. It outputs detection results in JSON format for further analysis.

## Prerequisites
- Python 3.8 or higher
- Required Python libraries:
  - `ultralytics` (Read the [ultralytics documentation](https://pypi.org/project/ultralytics/) for installation instructions)


## 🐍 Python
### Inference

- **On folder (With video/images)**:
  ``` bash
  python track_and_save.py --weights MV_yolo11x.pt --conf 0.3 --img-size 640 --source inference/
  ```

- **On video**:
  ``` bash
  python track_and_save.py --weights MV_yolo11x.pt --conf 0.3 --img-size 640 --source inference/videos/yourvideo.mp4
  ```

- **On image**:
  ``` bash
  python track_and_save.py --weights MV_yolo11x.pt --conf 0.3 --img-size 640 --source inference/images/yourimage.jpg
  ```

## 🚩 Flags and Arguments
- `--weights`: Path to the YOLO model weights file (Not specifying will download the weights file for Yolo11x from Ultralytics).
- `--source`: Path to the input file or folder.
- `--img-size`: Inference image size in pixels.
- `--conf-thres`: Object confidence threshold (default: `0.3`).
- `--iou-thres`: IOU threshold for non-max suppression (default: `0.45`).
- `--classes`: Filter by class (e.g., `--classes 0 2 3`, classes from the MicroVision dataset can be found in [MicroVision.yaml](MicroVision.yaml)).
- `--agnostic-nms`: Use class-agnostic NMS.
- `--augment`: Enable augmented inference.
- `--project`: Directory to save results (default: `runs/detect`).
- `--name`: Subfolder name for results (default: `exp`).
- `--save`: Save the video output.

## ✨ Models

| Model                                                                                | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>T4 TensorRT10<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| ------------------------------------------------------------------------------------ | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| [MV_YOLO11n](https://github.com/Rahul-Pi/MicroVision/yolo11n.pt) | 640                   | 39.5                 | 56.1 ± 0.8                     | 1.5 ± 0.0                           | 2.6                | 6.5               |
| [MV_YOLO11s](https://github.com/Rahul-Pi/MicroVision/yolo11s.pt) | 640                   | 47.0                 | 90.0 ± 1.2                     | 2.5 ± 0.0                           | 9.4                | 21.5              |
| [MV_YOLO11m](https://github.com/Rahul-Pi/MicroVision/yolo11m.pt) | 640                   | 51.5                 | 183.2 ± 2.0                    | 4.7 ± 0.1                           | 20.1               | 68.0              |
| [MV_YOLO11l](https://github.com/Rahul-Pi/MicroVision/yolo11l.pt) | 640                   | 53.4                 | 238.6 ± 1.4                    | 6.2 ± 0.1                           | 25.3               | 86.9              |
| [MV_YOLO11x](https://github.com/Rahul-Pi/MicroVision/yolo11x.pt) | 640                   | 54.7                 | 462.8 ± 6.7                    | 11.3 ± 0.2                          | 56.9               | 194.9             |

- **Speed** metrics are averaged over val images using an NVIDIA A100 instance.