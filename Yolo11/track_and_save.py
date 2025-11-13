import json
from ultralytics import YOLO
import argparse
import os
from pathlib import Path
 
def load_files(source_path):
    # If the input is a folder, return an array of all video files in the folder else retrurn the video or image path as an array
    if os.path.isdir(source_path):
        p = Path(source_path)
        videos, images = [], []
        for ext in ["jpg", "jpeg", "png", "bmp"]:
            for image in p.glob(f"*.{ext}"):
                images.append(str(image))
        for ext in ["mp4", "avi", "mov", "mkv", "flv"]:
            for video in p.glob(f"*.{ext}"):
                videos.append(str(video))
        return videos, images
    else:
        # if the input is a video or image file specify the path in the corresponding order
        if source_path.endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv')):
            return [source_path], []
        elif source_path.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            return [], [source_path]
        else:
            raise ValueError("Unsupported file type. Please provide a video or image file.")

def run_inference_on_image(image_path, model_path, output_json, args):
    vid_meta = {"filename": Path(image_path).stem, "detection": []}
    model = YOLO(model_path)
    results = model.predict(source=image_path, conf=args.conf_thres, iou=args.iou_thres,
                             classes=args.classes, imgsz=args.img_size,
                             augment=args.augment, agnostic_nms=args.agnostic_nms)
    detection = {"frame": 0, "objects": []}
    for box in results[0].boxes:
        coords = box.xyxy[0].tolist()
        class_id = int(box.cls[0].item())
        obj = {"obj_id": None, "category_id": class_id, "bbox": coords}
        detection["objects"].append(obj)
    vid_meta["detection"].append(detection)
    with open(output_json, "w") as f:
        json.dump(vid_meta, f, indent=4)


def run_inference_on_video(video_path, model_path, output_json, args):
    vid_meta = dict()
    vid_meta["filename"] = Path(video_path).stem
    vid_meta["detection"] = []
 
    # Load the YOLO model
    model = YOLO(model_path)
 
    # Use the model's 'track' mode to get tracking IDs for objects
    # This returns results with track IDs for each box (box.id)
    results = model.track(source=video_path, conf=args.conf_thres, iou=args.iou_thres,
                            classes=args.classes, imgsz=args.img_size,
                            augment=args.augment, agnostic_nms=args.agnostic_nms,
                            save=args.save, project=args.project, name=args.name,
                            save_txt=False, save_frames=False, show=False,
                            tracker="botsort.yaml")
 
    # Collect bounding box and tracking data for each frame
    for frame_index, result in enumerate(results):
        detection = dict()
        detection["frame"] = frame_index
        detection["objects"] = []
 
        for box in result.boxes:
            coords = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
            class_id = int(box.cls[0].item())
            # box.id is the tracker-assigned ID for each object
            track_id = (
                int(box.id[0].item())
                if box.id is not None
                else None
            )
            if track_id is None:
                continue
 
            obj = dict()
            # 'obj_id' becomes the tracker-provided ID
            obj["obj_id"] = track_id
            # 'category_id' remains the YOLO class ID
            obj["category_id"] = class_id
            obj["bbox"] = coords
            detection["objects"].append(obj)
 
        vid_meta["detection"].append(detection)
 
    # Save the JSON
    with open(output_json, "w") as f:
        json.dump(vid_meta, f, indent=4)
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolo11x.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--save', action='store_true', help='save the video output')
    args = parser.parse_args()
   
    source_path = args.source
    model_path = args.weights
    output_save_path = os.path.join(args.project, args.name)
    os.makedirs(output_save_path, exist_ok=True)
    
    videos, images = load_files(source_path)

    if len(images) > 0:
        for source_file in images:
            output_json = os.path.join(output_save_path, Path(source_file).stem + ".json")
            run_inference_on_image(source_file, model_path, output_json, args)
   
    if len(videos) > 0:
        for source_file in videos:
            output_json = os.path.join(output_save_path, Path(source_file).stem + ".json")
            run_inference_on_video(source_file, model_path, output_json, args)