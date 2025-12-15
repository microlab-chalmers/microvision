import os
import glob
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import cohen_kappa_score

def yolo_to_coco_bbox(yolo_bbox, img_width, img_height):
    """
    Converts a YOLO format bounding box [x_c, y_c, w, h] (normalized)
    to a COCO format bounding box [x_min, y_min, w, h] (absolute pixels).
    """
    x_c, y_c, w, h = yolo_bbox
    x_c_abs = x_c * img_width
    y_c_abs = y_c * img_height
    w_abs = w * img_width
    h_abs = h * img_height
    x_min = x_c_abs - (w_abs / 2)
    y_min = y_c_abs - (h_abs / 2)
    return [x_min, y_min, w_abs, h_abs]

def calculate_iou(box1, box2):
    """
    Calculates IoU for two boxes in [x_min, y_min, w, h] format.
    """
    x1, y1, w1, h1 = box1
    x1_max, y1_max = x1 + w1, y1 + h1
    x2, y2, w2, h2 = box2
    x2_max, y2_max = x2 + w2, y2 + h2

    inter_x_min = max(x1, x2)
    inter_y_min = max(y1, y2)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_w = max(0, inter_x_max - inter_x_min)
    inter_h = max(0, inter_y_max - inter_y_min)
    inter_area = inter_w * inter_h

    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area

    iou = inter_area / (union_area + 1e-6)
    return iou

def load_boxes_from_file(txt_file, img_width, img_height):
    """
    Loads all boxes from a YOLO txt file.
    Returns list of (class_id, bbox, area)
    """
    boxes = []
    if not os.path.exists(txt_file):
        return boxes

    with open(txt_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            
            class_id = int(parts[0])
            yolo_bbox = [float(p) for p in parts[1:5]]
            coco_bbox = yolo_to_coco_bbox(yolo_bbox, img_width, img_height)
            area = coco_bbox[2] * coco_bbox[3]
            
            boxes.append((class_id, coco_bbox, area))
    return boxes

def categorize_area(area, area_ranges):
    """Categorizes an area into 'small', 'medium', 'large', or 'all'."""
    if area < area_ranges[1][1]:
        return 'small'
    elif area < area_ranges[2][1]:
        return 'medium'
    else:
        return 'large'

def compute_pairwise_agreement(annotator1_path, annotator2_path, img_width, img_height, area_ranges):
    """
    Compares two annotators and returns kappa and IoU metrics by size category and class.
    Returns: {
        'all': {'kappa': float, 'iou': float, 'matches': int, 'total': int},
        'small': {...},
        'medium': {...},
        'large': {...},
        'by_class': {class_id: {'all': {...}, 'small': {...}, ...}, ...}
    }
    """
    IOU_THRESHOLD = 0.5
    
    metrics_by_size = {
        'all': {'matches': 0, 'disagreements': 0, 'ious': []},
        'small': {'matches': 0, 'disagreements': 0, 'ious': []},
        'medium': {'matches': 0, 'disagreements': 0, 'ious': []},
        'large': {'matches': 0, 'disagreements': 0, 'ious': []}
    }
    
    # Also track by class
    metrics_by_class = {}
    
    ann1_files = sorted(glob.glob(os.path.join(annotator1_path, '*.txt')))
    
    for ann1_txt_file in ann1_files:
        base_filename = os.path.basename(ann1_txt_file)
        ann2_txt_file = os.path.join(annotator2_path, base_filename)
        
        ann1_boxes = load_boxes_from_file(ann1_txt_file, img_width, img_height)
        ann2_boxes = load_boxes_from_file(ann2_txt_file, img_width, img_height)
        
        num_ann1 = len(ann1_boxes)
        num_ann2 = len(ann2_boxes)
        
        if num_ann1 > 0 and num_ann2 > 0:
            # Build IoU matrix (only compare same class)
            iou_matrix = np.zeros((num_ann1, num_ann2))
            for i in range(num_ann1):
                class_i, box_i, area_i = ann1_boxes[i]
                for j in range(num_ann2):
                    class_j, box_j, area_j = ann2_boxes[j]
                    # Only match same class
                    if class_i == class_j:
                        iou_matrix[i, j] = calculate_iou(box_i, box_j)
            
            # Hungarian algorithm for matching
            cost_matrix = 1.0 - iou_matrix
            cost_matrix[iou_matrix < IOU_THRESHOLD] = 100.0
            
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            matched_pairs = []
            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] < 1.0:
                    iou = iou_matrix[r, c]
                    class_i, box_i, area_i = ann1_boxes[r]
                    class_j, box_j, area_j = ann2_boxes[c]
                    avg_area = (area_i + area_j) / 2.0
                    matched_pairs.append((class_i, iou, avg_area))
            
            # Count matches and disagreements
            matched_ann1 = set(r for r, c in zip(row_ind, col_ind) if cost_matrix[r, c] < 1.0)
            matched_ann2 = set(c for r, c in zip(row_ind, col_ind) if cost_matrix[r, c] < 1.0)
            
            # Update metrics for matched boxes
            for class_id, iou, avg_area in matched_pairs:
                size_cat = categorize_area(avg_area, area_ranges)
                
                # Overall metrics
                metrics_by_size['all']['matches'] += 1
                metrics_by_size['all']['ious'].append(iou)
                metrics_by_size[size_cat]['matches'] += 1
                metrics_by_size[size_cat]['ious'].append(iou)
                
                # Per-class metrics
                if class_id not in metrics_by_class:
                    metrics_by_class[class_id] = {
                        'all': {'matches': 0, 'disagreements': 0, 'ious': []},
                        'small': {'matches': 0, 'disagreements': 0, 'ious': []},
                        'medium': {'matches': 0, 'disagreements': 0, 'ious': []},
                        'large': {'matches': 0, 'disagreements': 0, 'ious': []}
                    }
                metrics_by_class[class_id]['all']['matches'] += 1
                metrics_by_class[class_id]['all']['ious'].append(iou)
                metrics_by_class[class_id][size_cat]['matches'] += 1
                metrics_by_class[class_id][size_cat]['ious'].append(iou)
            
            # Count unmatched boxes as disagreements
            for i in range(num_ann1):
                if i not in matched_ann1:
                    class_id, _, area = ann1_boxes[i]
                    size_cat = categorize_area(area, area_ranges)
                    metrics_by_size['all']['disagreements'] += 1
                    metrics_by_size[size_cat]['disagreements'] += 1
                    
                    if class_id not in metrics_by_class:
                        metrics_by_class[class_id] = {
                            'all': {'matches': 0, 'disagreements': 0, 'ious': []},
                            'small': {'matches': 0, 'disagreements': 0, 'ious': []},
                            'medium': {'matches': 0, 'disagreements': 0, 'ious': []},
                            'large': {'matches': 0, 'disagreements': 0, 'ious': []}
                        }
                    metrics_by_class[class_id]['all']['disagreements'] += 1
                    metrics_by_class[class_id][size_cat]['disagreements'] += 1
            
            for j in range(num_ann2):
                if j not in matched_ann2:
                    class_id, _, area = ann2_boxes[j]
                    size_cat = categorize_area(area, area_ranges)
                    metrics_by_size['all']['disagreements'] += 1
                    metrics_by_size[size_cat]['disagreements'] += 1
                    
                    if class_id not in metrics_by_class:
                        metrics_by_class[class_id] = {
                            'all': {'matches': 0, 'disagreements': 0, 'ious': []},
                            'small': {'matches': 0, 'disagreements': 0, 'ious': []},
                            'medium': {'matches': 0, 'disagreements': 0, 'ious': []},
                            'large': {'matches': 0, 'disagreements': 0, 'ious': []}
                        }
                    metrics_by_class[class_id]['all']['disagreements'] += 1
                    metrics_by_class[class_id][size_cat]['disagreements'] += 1
        else:
            # All boxes are unmatched
            for class_id, _, area in ann1_boxes:
                size_cat = categorize_area(area, area_ranges)
                metrics_by_size['all']['disagreements'] += 1
                metrics_by_size[size_cat]['disagreements'] += 1
                
                if class_id not in metrics_by_class:
                    metrics_by_class[class_id] = {
                        'all': {'matches': 0, 'disagreements': 0, 'ious': []},
                        'small': {'matches': 0, 'disagreements': 0, 'ious': []},
                        'medium': {'matches': 0, 'disagreements': 0, 'ious': []},
                        'large': {'matches': 0, 'disagreements': 0, 'ious': []}
                    }
                metrics_by_class[class_id]['all']['disagreements'] += 1
                metrics_by_class[class_id][size_cat]['disagreements'] += 1
            
            for class_id, _, area in ann2_boxes:
                size_cat = categorize_area(area, area_ranges)
                metrics_by_size['all']['disagreements'] += 1
                metrics_by_size[size_cat]['disagreements'] += 1
                
                if class_id not in metrics_by_class:
                    metrics_by_class[class_id] = {
                        'all': {'matches': 0, 'disagreements': 0, 'ious': []},
                        'small': {'matches': 0, 'disagreements': 0, 'ious': []},
                        'medium': {'matches': 0, 'disagreements': 0, 'ious': []},
                        'large': {'matches': 0, 'disagreements': 0, 'ious': []}
                    }
                metrics_by_class[class_id]['all']['disagreements'] += 1
                metrics_by_class[class_id][size_cat]['disagreements'] += 1
    
    # Calculate kappa and average IoU for each size category
    results = {}
    for size_cat in ['all', 'small', 'medium', 'large']:
        matches = metrics_by_size[size_cat]['matches']
        disagreements = metrics_by_size[size_cat]['disagreements']
        total = matches + disagreements
        
        # Cohen's kappa: matches / total
        kappa = matches / total if total > 0 else 0.0
        
        # Average IoU
        ious = metrics_by_size[size_cat]['ious']
        avg_iou = np.mean(ious) if len(ious) > 0 else 0.0
        
        results[size_cat] = {
            'kappa': kappa,
            'iou': avg_iou,
            'matches': matches,
            'total': total
        }
    
    # Calculate per-class results
    results['by_class'] = {}
    for class_id in sorted(metrics_by_class.keys()):
        results['by_class'][class_id] = {}
        for size_cat in ['all', 'small', 'medium', 'large']:
            matches = metrics_by_class[class_id][size_cat]['matches']
            disagreements = metrics_by_class[class_id][size_cat]['disagreements']
            total = matches + disagreements
            
            kappa = matches / total if total > 0 else 0.0
            ious = metrics_by_class[class_id][size_cat]['ious']
            avg_iou = np.mean(ious) if len(ious) > 0 else 0.0
            
            results['by_class'][class_id][size_cat] = {
                'kappa': kappa,
                'iou': avg_iou,
                'matches': matches,
                'total': total
            }
    
    return results

def print_agreement_summary(all_kappas, all_ious, categories_dict):
    """
    Prints a table with kappa and IoU metrics by class and size.
    """
    print("\n" + "="*120)
    print("  INTER-RATER AGREEMENT SUMMARY")
    print("="*120)
    
    # Header
    header = "Class".ljust(20)
    header += "small_kappa".ljust(15) + "medium_kappa".ljust(15) + "large_kappa".ljust(15) + "ALL_kappa".ljust(15)
    header += "small_IoU".ljust(12) + "medium_IoU".ljust(12) + "large_IoU".ljust(12) + "ALL_IoU".ljust(12)
    print(header)
    print("-" * 120)
    
    # Get all classes
    all_classes = sorted(categories_dict.keys())
    
    for class_id in all_classes:
        class_name = categories_dict[class_id]
        row = class_name.ljust(20)
        
        # Kappas for each size
        row += f"{all_kappas[class_id]['small']['average']:.3f}".ljust(15)
        row += f"{all_kappas[class_id]['medium']['average']:.3f}".ljust(15)
        row += f"{all_kappas[class_id]['large']['average']:.3f}".ljust(15)
        row += f"{all_kappas[class_id]['all']['average']:.3f}".ljust(15)
        
        # IoUs for each size
        row += f"{all_ious[class_id]['small']['average']:.3f}".ljust(12)
        row += f"{all_ious[class_id]['medium']['average']:.3f}".ljust(12)
        row += f"{all_ious[class_id]['large']['average']:.3f}".ljust(12)
        row += f"{all_ious[class_id]['all']['average']:.3f}".ljust(12)
        
        print(row)
    
    # All row (average across classes)
    row = "All".ljust(20)
    row += f"{all_kappas['all']['small']['average']:.3f}".ljust(15)
    row += f"{all_kappas['all']['medium']['average']:.3f}".ljust(15)
    row += f"{all_kappas['all']['large']['average']:.3f}".ljust(15)
    row += f"{all_kappas['all']['all']['average']:.3f}".ljust(15)
    row += f"{all_ious['all']['small']['average']:.3f}".ljust(12)
    row += f"{all_ious['all']['medium']['average']:.3f}".ljust(12)
    row += f"{all_ious['all']['large']['average']:.3f}".ljust(12)
    row += f"{all_ious['all']['all']['average']:.3f}".ljust(12)
    print(row)


# --- SCRIPT CONFIGURATION ---
if __name__ == "__main__":
    
    # --- 1. Define your annotators ---
    ANNOTATORS = ["a1", "a2", "a3"]
    
    # --- 2. Define your paths ---
    BASE_DIR = "."
    
    # --- 3. Define image properties ---
    IMG_WIDTH = 1920
    IMG_HEIGHT = 1080
    
    # --- 4. Define categories ---
    CATEGORIES = [
        {"id": 0, "name": "pedestrian", "supercategory": "none"},
        {"id": 1, "name": "bicycle", "supercategory": "none"},
        {"id": 2, "name": "cyclist", "supercategory": "none"},
        {"id": 3, "name": "e-scooter", "supercategory": "none"},
        {"id": 4, "name": "e-scooterist", "supercategory": "none"}
    ]
    CATEGORIES_DICT = {c['id']: c['name'] for c in CATEGORIES}
    
    # --- 5. Define your custom area ranges (in pixels squared) ---
    s_max = 96**2   # 9216
    m_max = 288**2  # 82944
    l_min = m_max
    
    AREA_RANGES = [
        [0, 1e10],          # all (index 0)
        [0, s_max],         # small (index 1)
        [s_max, m_max],     # medium (index 2)
        [l_min, 1e10]       # large (index 3)
    ]
    
    # --- 6. Compute all pairwise agreements ---
    print("Computing pairwise agreements...")
    
    pairs = [
        (ANNOTATORS[0], ANNOTATORS[1]),  # a1 vs a2
        (ANNOTATORS[0], ANNOTATORS[2]),  # a1 vs a3
        (ANNOTATORS[1], ANNOTATORS[2])   # a2 vs a3
    ]
    
    pairwise_results = {}
    
    for ann1, ann2 in pairs:
        pair_name = f"{ann1} vs {ann2}"
        ann1_path = os.path.join(BASE_DIR, ann1)
        ann2_path = os.path.join(BASE_DIR, ann2)
        
        results = compute_pairwise_agreement(ann1_path, ann2_path, IMG_WIDTH, IMG_HEIGHT, AREA_RANGES)
        pairwise_results[pair_name] = results
        print(f"  Computed {pair_name}")
    
    # --- 7. Calculate averages across the three pairs, by class and overall ---
    avg_kappas = {
        'all': {}  # For overall metrics (not per-class)
    }
    avg_ious = {
        'all': {}  # For overall metrics (not per-class)
    }
    
    # First, initialize per-class dictionaries
    for class_id in CATEGORIES_DICT.keys():
        avg_kappas[class_id] = {}
        avg_ious[class_id] = {}
    
    for size_cat in ['all', 'small', 'medium', 'large']:
        # Overall metrics (across all classes)
        kappas = [pairwise_results[pair][size_cat]['kappa'] for pair in pairwise_results]
        ious = [pairwise_results[pair][size_cat]['iou'] for pair in pairwise_results]
        
        avg_kappas['all'][size_cat] = {
            'a1 vs a2': pairwise_results['a1 vs a2'][size_cat]['kappa'],
            'a1 vs a3': pairwise_results['a1 vs a3'][size_cat]['kappa'],
            'a2 vs a3': pairwise_results['a2 vs a3'][size_cat]['kappa'],
            'average': np.mean(kappas)
        }
        
        avg_ious['all'][size_cat] = {
            'a1 vs a2': pairwise_results['a1 vs a2'][size_cat]['iou'],
            'a1 vs a3': pairwise_results['a1 vs a3'][size_cat]['iou'],
            'a2 vs a3': pairwise_results['a2 vs a3'][size_cat]['iou'],
            'average': np.mean(ious)
        }
        
        # Per-class metrics
        for class_id in CATEGORIES_DICT.keys():
            class_kappas = []
            class_ious = []
            for pair in pairwise_results:
                if class_id in pairwise_results[pair]['by_class']:
                    class_kappas.append(pairwise_results[pair]['by_class'][class_id][size_cat]['kappa'])
                    class_ious.append(pairwise_results[pair]['by_class'][class_id][size_cat]['iou'])
            
            if class_kappas:
                avg_kappas[class_id][size_cat] = {
                    'average': np.mean(class_kappas)
                }
                avg_ious[class_id][size_cat] = {
                    'average': np.mean(class_ious)
                }
            else:
                avg_kappas[class_id][size_cat] = {'average': 0.0}
                avg_ious[class_id][size_cat] = {'average': 0.0}
    
    # --- 8. Print the summary table ---
    print_agreement_summary(avg_kappas, avg_ious, CATEGORIES_DICT)
