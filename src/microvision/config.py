IMG_WIDTH = 1920
"""Image width (pixels)"""

IMG_HEIGHT = 1080
"""Image height (pixels)"""

OBJECT_CLASSES = {
    0: "pedestrian",
    1: "bicycle",
    2: "cyclist",
    3: "e-scooter",
    4: "e-scooterist",
}
"""Object class index and name"""

CMAP_TAB10 = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
"""Color map for objects (tab10)"""

AREA_THSD = {"small": 96**2, "medium": 288**2}
"""Area thresholds for object sizes (pixels^2)"""
