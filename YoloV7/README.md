
# Yolov7 Detect and Track
This script uses YOLOv7 for object detection and tracking on videos and images. It outputs detection results in JSON format for further analysis.

## Prerequisites
- Python 3.8 or higher
- Required Python libraries:
  - `torch`
  - `numpy`
  - `opencv-python`
  - `json`
  - `subprocess`
- Additional Tools:
  - FFmpeg: Required for audio processing when using the `--add-audio` flag. Install it from [FFmpeg.org](https://ffmpeg.org).

## 🐍 Python
### Inference
- **Detect objects in a folder of images/videos**:
  ```bash
  python detect_or_track.py --weights yolov7x.pt --source inference/ --img-size 640 --conf-thres 0.5
  ```

- **Detect objects in a single video**:
  ```bash
  python detect_or_track.py --weights yolov7x.pt --source inference/videos/video.mp4 --track
  ```

- **Run object tracking with a tracking line at the center of the bounding box**:
  ```bash
  python3 detect_or_track.py --weight yolov7x.pt --conf 0.6 --classes 0 1 2 --img-size 640 --source yourfile.mp4 --show-track
  ```

- **Run object tracking without adding a tracking line**:
  ```bash
  python3 detect_or_track.py --weight yolov7x.pt --conf 0.6 --classes 0 1 2 --img-size 640 --source yourfile.mp4 --track
  ```

- **Obtain tracking at the bottom of the bounding box**:
  ```bash
  python3 detect_or_track.py --weight yolov7x.pt --conf 0.6 --classes 0 1 2 --img-size 640 --source yourfile.mp4 --track --base-track
  ```

- **Save annotations in a JSON file**:
  ```bash
  python3 detect_or_track.py --weight yolov7x.pt --conf 0.6 --classes 0 --img-size 640 --source yourfile.mp4 --base-track --store-meta
  ```

- **Anonymize the video**:
  ```bash
  python3 detect_or_track.py --weight yolov7x.pt --conf 0.6 --classes 0 2 --img-size 640 --source yourfile.mp4 --blurbox
  ```

- **Retain the audio of the original video**:
  ```bash
  python3 detect_or_track.py --weight yolov7x.pt --conf 0.6 --classes 0 2 --img-size 640 --source yourfile.mp4 --blurbox --add-audio
  ```


## 🚩 Flags and Arguments

### General Arguments

- `--weights`: Path to the YOLO model weights file (default: `yolov7x.pt`).
- `--source`: Path to the input file or folder (default: `inference/images`).
- `--img-size`: Inference image size in pixels (default: `640`).
- `--conf-thres`: Object confidence threshold (default: `0.5`).
- `--iou-thres`: IOU threshold for non-max suppression (default: `0.45`).
- `--device`: Specify the device to use (`cpu` or `cuda:0`, etc.).
- `--project`: Directory to save results (default: `runs/detect`).
- `--name`: Subfolder name for results (default: `exp`).
- `--exist-ok`: Allow overwriting existing results without incrementing the folder name.
- `--no-trace`: Disable model tracing for inference.

### Display and Output

- `--view-img`: Display results in a window.
- `--save-txt`: Save detection results to `.txt` files.
- `--save-conf`: Save confidence scores in `.txt` labels.
- `--nosave`: Do not save images/videos.
- `--show-fps`: Display FPS on the output video.

### Detection and Tracking

- `--classes`: Filter detections by class (e.g., `--classes 0 1 2`).
- `--agnostic-nms`: Perform class-agnostic NMS.
- `--augment`: Enable augmented inference.
- `--track`: Enable object tracking.
- `--base-track`: Use base tracking mode.
- `--show-track`: Display tracked paths on the output video.

### Bounding Box Customization

- `--thickness`: Set the thickness of bounding boxes and labels (default: `2`).
- `--nobbox`: Do not display bounding boxes.
- `--blackbox`: Add a black box to anonymize bounding boxes.
- `--faceonlybox`: Add a black box to anonymize only the face region.
- `--blurbox`: Blur the bounding box region.
- `--nolabel`: Do not display labels on bounding boxes.
- `--unique-track-color`: Assign a unique color to each tracked object.

### Metadata and Audio

- `--store-meta`: Store metadata for detections in a JSON file.
- `--add-audio`: Retain the audio from the input video in the output.

## ✨ Models

| Model                                                                                        | size<br><sup>(pixels) | mAP<sup>box<br>50-95 | mAP<sup>mask<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>T4 TensorRT10<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| -------------------------------------------------------------------------------------------- | --------------------- | -------------------- | --------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| [MV_YOLOv7n](https://github.com/Rahul-Pi/MicroVision/yolo7n.pt) | 640                   | 38.9                 | 32.0                  | 65.9 ± 1.1                     | 1.8 ± 0.0                           | 2.9                | 10.4              |
| [MV_YOLOv7s](https://github.com/Rahul-Pi/MicroVision/yolo7s.pt) | 640                   | 46.6                 | 37.8                  | 117.6 ± 4.9                    | 2.9 ± 0.0                           | 10.1               | 35.5              |
| [MV_YOLOv7m](https://github.com/Rahul-Pi/MicroVision/yolo7m.pt) | 640                   | 51.5                 | 41.5                  | 281.6 ± 1.2                    | 6.3 ± 0.1                           | 22.4               | 123.3             |
| [MV_YOLOv7l](https://github.com/Rahul-Pi/MicroVision/yolo7l.pt) | 640                   | 53.4                 | 42.9                  | 344.2 ± 3.2                    | 7.8 ± 0.2                           | 27.6               | 142.2             |
| [MV_YOLOv7x](https://github.com/Rahul-Pi/MicroVision/yolo7x.pt) | 640                   | 54.7                 | 43.8                  | 664.5 ± 3.2                    | 15.8 ± 0.7                          | 62.1               | 319.0             |

- **Speed** metrics are averaged over val images using an NVIDIA A100 instance.