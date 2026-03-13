from pathlib import Path

from PIL import Image
from supervision import BoxAnnotator, ColorPalette, Position, RichLabelAnnotator

from microvision.config import CMAP_TAB10
from microvision.labels import anno2detections

palette = ColorPalette.from_hex(CMAP_TAB10)

box_annotator = BoxAnnotator(color=palette, thickness=16)

label_annotator = RichLabelAnnotator(
    color=palette,
    text_position=Position.TOP_LEFT,
    font_size=100,
)


def annotate(img, det, labels=None):
    """_summary_

    Args:
        img (PIL.Image): _description_
        det (_type_): _description_
        labels (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    img = box_annotator.annotate(img, det)

    # Annotate labels if supplied
    if labels:
        img = label_annotator.annotate(img, det, labels)

    return img


def plot_annotations(path_data, img_name, nms=False, path_out=None):

    path_img = Path(path_data, "images", f"{img_name}.jpg")
    path_labels = Path(path_data, "labels", f"{img_name}.txt")

    img = Image.open(path_img)

    if path_labels.exists():
        det, labels = anno2detections(path_labels)
        img = annotate(img, det, labels)
    else:
        img = annotate(img, det)

    if nms:
        det = det.with_nms(threshold=0.5, class_agnostic=False)

    if path_out is not None:
        Path(path_out).parent.mkdir(exist_ok=True)
        img.save(path_out)
    else:
        img.show()


if __name__ == "__main__":
    plot_annotations(
        path_data=r"C:\Users\Alexander Rasch\Desktop\test",
        img_name="S0007_001",
        path_out="results/plots/S0007_001.png",
    )
