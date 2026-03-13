import argparse
import shutil
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def split_data(
    path_data: str,
    path_meta: str,
    path_out: str,
    folder_images: str = "images",
    split_config: str = "split",
    override: bool = False,
) -> None:
    """Split data (images, labels) into train/val/test sets

    Args:
        path_data (str): Path to dataset (containing the folders `images` and `labels`).
        path_meta (str): Path to the meta data file (`meta.csv`).
        path_out (str): Path to the output directory where the splits will be placed.
        folder_images (str, optional): Name of image folders (if different). Defaults to "images".
        split_config (str, optional): Which split configuration to apply. Defaults to "split".
        override (bool, optional): Whether to override existing data or not (will delete exisiting data first). Defaults to False.
    """
    # Set up paths to txt files that include the image paths
    meta = pd.read_csv(path_meta)

    path_data = Path(path_data)

    splits = meta[split_config].unique()
    print(f"Available splits: {splits}")

    print(meta[split_config].value_counts())

    print(f"Creating {path_out}")
    path_out = Path(path_out)
    path_out.mkdir(exist_ok=override)

    # Remove existing folders if override
    if override and path_out.is_dir():
        shutil.rmtree(path_out)
        path_out.mkdir()

    for split in splits:
        print(f"Working with {split}")

        # Set up output-folder paths
        _folder_split = Path(path_out, split)
        _folder_images = Path(path_out, split, "images")
        _folder_labels = Path(path_out, split, "labels")

        # Create folders if not existing
        for _f in [_folder_split, _folder_images, _folder_labels]:
            _f.mkdir()

        meta_split = meta.loc[meta[split_config] == split]
        for _, row in tqdm(meta_split.iterrows(), total=meta_split.shape[0]):
            frame = row["img"]
            _path_image = list(
                Path(path_data, folder_images).glob(f"*{frame}.[jpg][jpeg]*")
            )
            if len(_path_image) == 1:
                _path_image = _path_image[0]
            else:
                # raise LookupError(f"More than one or no image found ({frame})")
                continue

            _path_label = Path(path_data, "labels", frame + ".txt")

            shutil.copy2(_path_image, _folder_images)
            if _path_label.exists():
                shutil.copy2(_path_label, _folder_labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="MicroVision split data into train/val/test according to meta.csv"
    )

    parser.add_argument("--path_data", help="folder for training data")
    parser.add_argument("--path_meta", help="video directory")
    parser.add_argument("--path_out", help="json directory")
    parser.add_argument("--split_config", default="split", help="json directory")
    parser.add_argument(
        "--override", default=False, action="store_true", help="Delete existing folders"
    )

    args = parser.parse_args()

    split_data(
        path_data=args.path_data,
        path_meta=args.path_meta,
        path_out=args.path_out,
        split_config=args.split_config,
        override=args.override,
    )

    print("Done.")
