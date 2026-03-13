import argparse

from microvision.labels import yolo2coco

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="MicroVision convert annotations from YOLO to COCO JSON format"
    )

    parser.add_argument(
        "--path_yolo",
        help="Folder with YOLO annotations (needs to contain an images and a labels folder)",
    )
    parser.add_argument("--path_json", help="Output file (JSON)")

    args = parser.parse_args()

    yolo2coco(path_yolo=args.path_yolo, path_json=args.path_json)

    print("Done.")
