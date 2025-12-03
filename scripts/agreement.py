import os
import glob
import numpy as np
import pandas as pd
from itertools import combinations
from pathlib import Path

# =============================================================
# CONFIGURATION
# =============================================================
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080

ANNOTATOR_DIRS = {
    "ann1": "data/agreement/a1",
    "ann2": "data/agreement/a2",
    "ann3": "data/agreement/a3",
}

IOU_MATCH_THRESHOLD = 0.5  # IoU threshold for matching boxes
SIZE_THRESHOLDS_PIXELS = {
    "small": 96*96,        # e.g., 32x32 px
    "medium": 288*288,     # e.g., 128x128 px
    "large": np.inf         # anything bigger
}

CLASSES = [0, 1, 2, 3, 4]
CLASS_NAMES = {
    0: "pedestrian",
    1: "bicycle",
    2: "cyclist",
    3: "e-scooter",
    4: "e-scooterist"
}

SIZES = ["small", "medium", "large"]

# =============================================================
# HELPER FUNCTIONS
# =============================================================
def load_yolo_annotations(txt_path):
    if not os.path.exists(txt_path):
        return np.zeros((0,5))
    data = []
    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, cx, cy, w, h = map(float, parts)
            data.append([int(cls), cx, cy, w, h])
    return np.array(data)

def yolo_to_bbox(box):
    cls, cx, cy, w, h = box
    x_c = cx * IMAGE_WIDTH
    y_c = cy * IMAGE_HEIGHT
    bw = w * IMAGE_WIDTH
    bh = h * IMAGE_HEIGHT
    x1 = x_c - bw / 2
    y1 = y_c - bh / 2
    x2 = x_c + bw / 2
    y2 = y_c + bh / 2
    return int(cls), x1, y1, x2, y2

def iou(b1, b2):
    x1 = max(b1[1], b2[1])
    y1 = max(b1[2], b2[2])
    x2 = min(b1[3], b2[3])
    y2 = min(b1[4], b2[4])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0
    a1 = (b1[3]-b1[1])*(b1[4]-b1[2])
    a2 = (b2[3]-b2[1])*(b2[4]-b2[2])
    return inter / (a1 + a2 - inter)

def categorize_size_pixel(x1, y1, x2, y2):
    area = (x2 - x1) * (y2 - y1)
    for label, thr in SIZE_THRESHOLDS_PIXELS.items():
        if area <= thr:
            return label
    return "large"

def match_boxes_for_image(ann_boxes):
    pix = {ann: [yolo_to_bbox(b) for b in boxes] for ann, boxes in ann_boxes.items()}
    A, B, C = list(ann_boxes.keys())
    matches = []

    for idx_a, box_a in enumerate(pix[A]):
        best_b, best_b_iou = None, 0
        for idx_b, box_b in enumerate(pix[B]):
            iou_ab = iou(box_a, box_b)
            if iou_ab > best_b_iou:
                best_b = idx_b
                best_b_iou = iou_ab

        best_c, best_c_iou = None, 0
        for idx_c, box_c in enumerate(pix[C]):
            iou_ac = iou(box_a, box_c)
            if iou_ac > best_c_iou:
                best_c = idx_c
                best_c_iou = iou_ac

        if best_b_iou >= IOU_MATCH_THRESHOLD and best_c_iou >= IOU_MATCH_THRESHOLD:
            matches.append({A: pix[A][idx_a], B: pix[B][best_b], C: pix[C][best_c]})

    return matches

def krippendorff_alpha_nominal(data):
    categories = sorted({v for row in data for v in row if v is not None})
    cat_index = {c:i for i,c in enumerate(categories)}
    n_cat = len(categories)
    O = np.zeros((n_cat, n_cat))
    for row in data:
        vals = [v for v in row if v is not None]
        if len(vals) < 2:
            continue
        for i, j in combinations(vals, 2):
            O[cat_index[i], cat_index[j]] += 1
            O[cat_index[j], cat_index[i]] += 1
    Do = sum(O[i,j] for i in range(n_cat) for j in range(n_cat) if i!=j)
    m = O.sum(axis=1)
    De = sum(m[i]*m[j] for i in range(n_cat) for j in range(n_cat) if i!=j)
    if De == 0:
        return 1.0
    return 1 - Do/De

# =============================================================
# BUILD ALL MATCHES
# =============================================================
all_matches = []
images = sorted([os.path.basename(f) for f in glob.glob(os.path.join(ANNOTATOR_DIRS["ann1"], "*.txt"))])

for img in images:
    ann_boxes = {ann: load_yolo_annotations(os.path.join(d, img)) for ann, d in ANNOTATOR_DIRS.items()}
    matches = match_boxes_for_image(ann_boxes)
    for m in matches:
        cls_labels = []
        size_labels = []
        boxes = []
        for ann in ANNOTATOR_DIRS.keys():
            cls = m[ann][0]
            x1, y1, x2, y2 = m[ann][1], m[ann][2], m[ann][3], m[ann][4]
            size_labels.append(categorize_size_pixel(x1, y1, x2, y2))
            cls_labels.append(cls)
            boxes.append([x1, y1, x2, y2])
        pairwise_ious = [iou([0]+b1, [0]+b2) for b1,b2 in combinations(boxes,2)]
        mean_iou = np.mean(pairwise_ious) if pairwise_ious else np.nan
        all_matches.append({"image": img, "class_ann": cls_labels, "size_ann": size_labels, "mean_iou": mean_iou})

# =============================================================
# BUILD TABLE: ALPHA COLUMNS LEFT, MEAN IOU RIGHT
# =============================================================
rows = [CLASS_NAMES[c] for c in CLASSES] + ["ALL"]

alpha_cols = [f"{s}_alpha" for s in SIZES] + ["ALL_alpha"]
iou_cols = [f"{s}_IoU" for s in SIZES] + ["ALL_IoU"]
combined_table = pd.DataFrame(index=rows, columns=alpha_cols + iou_cols, dtype=float)

for cls in CLASSES:
    cls_name = CLASS_NAMES[cls]
    for size in SIZES:
        subset = [m["class_ann"] for m in all_matches if cls in m["class_ann"] and size in m["size_ann"]]
        subset_iou = [m["mean_iou"] for m in all_matches if cls in m["class_ann"] and size in m["size_ann"]]
        combined_table.at[cls_name, f"{size}_alpha"] = krippendorff_alpha_nominal(subset) if len(subset)>=2 else np.nan
        combined_table.at[cls_name, f"{size}_IoU"] = np.mean(subset_iou) if len(subset_iou)>=1 else np.nan

    subset_all_sizes = [m["class_ann"] for m in all_matches if cls in m["class_ann"]]
    subset_iou_all = [m["mean_iou"] for m in all_matches if cls in m["class_ann"]]
    combined_table.at[cls_name, "ALL_alpha"] = krippendorff_alpha_nominal(subset_all_sizes)
    combined_table.at[cls_name, "ALL_IoU"] = np.mean(subset_iou_all) if len(subset_iou_all)>=1 else np.nan

for size in SIZES:
    subset_alpha = [m["class_ann"] for m in all_matches if size in m["size_ann"]]
    subset_iou = [m["mean_iou"] for m in all_matches if size in m["size_ann"]]
    combined_table.at["ALL", f"{size}_alpha"] = krippendorff_alpha_nominal(subset_alpha) if len(subset_alpha)>=2 else np.nan
    combined_table.at["ALL", f"{size}_IoU"] = np.mean(subset_iou) if len(subset_iou)>=1 else np.nan

combined_table.at["ALL", "ALL_alpha"] = krippendorff_alpha_nominal([m["class_ann"] for m in all_matches])
combined_table.at["ALL", "ALL_IoU"] = np.mean([m["mean_iou"] for m in all_matches])

# =============================================================
# DISPLAY TABLE
# =============================================================
print("\n==================== Krippendorff's Alpha (left) + Mean IoU (right) ====================")
print(combined_table.round(3))
print("=======================================================================================")


combined_table.to_csv("annotator_agreement_alpha_left_iou_right.csv")



# =============================================================
# FIND SPECIFIC INSTANCES WITH LOW AGREEMENT
# =============================================================

def instance_disagreement_score(class_ann):
    """
    For nominal data, disagreement is simply:
    number of annotators NOT matching the majority class.
    """
    vals = [v for v in class_ann]
    majority = max(set(vals), key=vals.count)
    disag = sum(v != majority for v in vals)
    return disag  # 0 = perfect, 1-2 = some disagreement, 3 = total disagreement


# Build a detailed instance list
instance_records = []
for m in all_matches:
    classes = m["class_ann"]      # class labels from ann1, ann2, ann3
    sizes = m["size_ann"]         # size labels
    mean_iou = m["mean_iou"]
    image = m["image"]

    # per-instance disagreement score
    disagree_score = instance_disagreement_score(classes)

    # majority class
    majority_class = max(set(classes), key=classes.count)

    instance_records.append({
        "image": image,
        "classes": classes,
        "sizes": sizes,
        "mean_iou": mean_iou,
        "majority_class": majority_class,
        "disagree_score": disagree_score
    })

instance_df = pd.DataFrame(instance_records)





# =============================================================
# 1. Worst disagreement instances (sorted)
# =============================================================
worst_instances = instance_df.sort_values("disagree_score", ascending=False)


print("\n=================== TOP DISAGREEMENTS (all classes) ===================")
for idx, row in worst_instances.head(10).iterrows():
    print(f"- File: {row['image']}")
    print(f"  Annotator classes: {row['classes']}")
    print(f"  Size categories:  {row['sizes']}")
    print(f"  Mean IoU:         {row['mean_iou']:.2f}")
    print(f"  Disagreement:     {row['disagree_score']}")
    print("")


# =============================================================
# 2. Worst disagreement per class
# =============================================================
print("\n=================== WORST DISAGREEMENT PER CLASS ===================")
for cls in CLASSES:
    class_name = CLASS_NAMES[cls]
    sub = instance_df[instance_df["majority_class"] == cls]
    if len(sub) == 0:
        print(f"{class_name}: (no matched instances)")
        continue
    worst = sub.sort_values("disagree_score", ascending=False).iloc[0]
    print(f"\nClass: {class_name}")
    print(f"  File:             {worst['image']}")
    print(f"  Annotator labels: {worst['classes']}")
    print(f"  Size categories:  {worst['sizes']}")
    print(f"  Mean IoU:         {worst['mean_iou']:.2f}")
    print(f"  Disagreement:     {worst['disagree_score']}")


# =============================================================
# 3. Worst disagreement per (class × size)
# =============================================================
print("\n=================== WORST DISAGREEMENT PER CLASS × SIZE ===================")
for cls in CLASSES:
    for size in SIZES:
        class_name = CLASS_NAMES[cls]
        sub = instance_df[
            (instance_df["majority_class"] == cls) &
            (instance_df["sizes"].apply(lambda s: size in s))
        ]
        if len(sub) == 0:
            continue
        worst = sub.sort_values("disagree_score", ascending=False).iloc[0]
        print(f"\nClass: {class_name}, Size: {size}")
        print(f"  File:             {worst['image']}")
        print(f"  Annotator labels: {worst['classes']}")
        print(f"  Size categories:  {worst['sizes']}")
        print(f"  Mean IoU:         {worst['mean_iou']:.2f}")
        print(f"  Disagreement:     {worst['disagree_score']}")




import seaborn as sns
import matplotlib.pyplot as plt

# =============================================================
# SPLIT TABLE INTO ALPHA AND IOU
# =============================================================
alpha_cols = [c for c in combined_table.columns if "_alpha" in c]
iou_cols = [c for c in combined_table.columns if "_IoU" in c]

alpha_df = combined_table[alpha_cols].copy()
iou_df = combined_table[iou_cols].copy()

# =============================================================
# PLOT HEATMAPS
# =============================================================
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# Alpha heatmap
sns.heatmap(alpha_df, annot=True, fmt=".2f", cmap="Blues", cbar=True, ax=axes[0])
axes[0].set_title("Krippendorff's Alpha (Class Agreement)")
axes[0].set_ylabel("Class")
axes[0].set_xlabel("Size / Overall")

# IoU heatmap
sns.heatmap(iou_df, annot=True, fmt=".2f", cmap="Greens", cbar=True, ax=axes[1])
axes[1].set_title("Mean IoU (BBox Agreement)")
axes[1].set_ylabel("Class")
axes[1].set_xlabel("Size / Overall")

plt.tight_layout()
plt.show()
