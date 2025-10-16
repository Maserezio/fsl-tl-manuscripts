import xml.etree.ElementTree as ET
import numpy as np

def parse_xml_polygons(xml_path):
    """
    Parse TextLine polygons from a PAGE-style XML file.
    
    Returns:
        List of polygons: each polygon is a list of (x, y) tuples.
    """
    ns = {"ns": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"}
    tree = ET.parse(xml_path)
    root = tree.getroot()
    polygons = []
    for textline in root.findall(".//ns:TextLine", ns):
        coords = textline.find("ns:Coords", ns)
        if coords is not None and "points" in coords.attrib:
            points = []
            for xy in coords.attrib["points"].split():
                x, y = map(int, xy.split(","))
                points.append((x, y))
            polygons.append(points)
    return polygons

def polygon_iou(poly1, poly2):
    """
    Compute IoU between two polygons approximated as binary masks.
    
    poly1, poly2: list of (x, y) tuples
    Returns: IoU float
    """
    # Compute bounding box to create minimal mask
    xs = [x for x, y in poly1 + poly2]
    ys = [y for x, y in poly1 + poly2]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    
    width, height = max_x - min_x + 1, max_y - min_y + 1
    
    mask1 = np.zeros((height, width), dtype=np.uint8)
    mask2 = np.zeros((height, width), dtype=np.uint8)
    
    def fill_mask(poly, mask):
        from cv2 import fillPoly
        pts = np.array([[x - min_x, y - min_y] for x, y in poly], dtype=np.int32)
        fillPoly(mask, [pts], 1)
    
    fill_mask(poly1, mask1)
    fill_mask(poly2, mask2)
    
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    
    if union == 0:
        return 0.0
    return intersection / union

def polygon_to_mask(poly, mask_shape):
    """
    Convert a polygon to a binary mask.
    
    Parameters:
    - poly: list of (x, y) tuples defining the polygon
    - mask_shape: (height, width) tuple for the mask
    
    Returns:
    - Binary mask as numpy array
    """
    import cv2
    mask = np.zeros(mask_shape, dtype=np.uint8)
    if len(poly) < 3:  # Need at least 3 points for a polygon
        return mask
    
    pts = np.array(poly, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)

def polygon_to_baseline(poly):
    """
    Simple baseline approximation: fit a line through the bottom-most points.
    Returns a list of (x, y) points for baseline.
    """
    # Take min y at each x coordinate (approx bottom of polygon)
    poly_sorted = sorted(poly, key=lambda p: p[0])  # sort by x
    baseline = []
    for x, _ in poly_sorted:
        ys = [y for px, y in poly if px == x]
        if ys:
            baseline.append((x, max(ys)))  # bottom-most point
    return baseline
