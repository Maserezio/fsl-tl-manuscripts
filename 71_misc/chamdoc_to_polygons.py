"""
# ChamDOC
"""

import os
import cv2
import xml.etree.ElementTree as ET
import fnmatch
import re
from datetime import datetime
from pathlib import Path
import csv

# for each image in the dataset, generate PAGE XML with text line contours
dt_dir = "../datasets/CHAMDoc/Full-clean-Inscription/"
imgs = [f for f in os.listdir(dt_dir) if f.lower().endswith(('.png'))]

for img in imgs:
    orig_pic = os.path.join(dt_dir, img)
    mask_dir = "../datasets/CHAMDoc/GT_line_by_line/"
    file_name = os.path.basename(orig_pic)[:-4]

    # --- find matching masks ---
    matches = []
    for root_dir, dirnames, filenames in os.walk(mask_dir):
        for filename in fnmatch.filter(filenames, f"*{file_name}*"):
            matches.append(os.path.join(root_dir, filename))

    # --- natural sorting ---
    matches = sorted(matches, key=lambda x: [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', x)])

    # --- load original ---
    orig = cv2.imread(orig_pic)
    if orig is None:
        raise FileNotFoundError(orig_pic)
    height, width = orig.shape[:2]
    vis = orig.copy()

    # --- XML root without namespace ---
    root = ET.Element("PcGts", {
        "xmlns": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15",
        "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
        "xsi:schemaLocation": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15 "
                            "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15/pagecontent.xsd"
    })

    # Metadata
    metadata = ET.SubElement(root, "Metadata")
    ET.SubElement(metadata, "Creator").text = "Arthur XML Generator"
    ET.SubElement(metadata, "Created").text = datetime.now().isoformat()
    ET.SubElement(metadata, "LastChange").text = datetime.now().isoformat()

    # Page element
    page = ET.SubElement(root, "Page", {
        "imageFilename": os.path.basename(orig_pic),
        "imageWidth": str(width),
        "imageHeight": str(height)
    })

    # One text region
    region = ET.SubElement(page, "TextRegion", {"id": "region_textline", "custom": "0"})
    ET.SubElement(region, "Coords", {"points": f"0,0 {width},0 {width},{height} 0,{height}"})

    # --- process masks and generate TextLines with proper polygons ---
    for i, mask_path in enumerate(matches):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        # Ensure binary image
        _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # Apply morphological operations to connect text in lines
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
        dilated = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close, iterations=2)

        # Find contours of text lines
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            continue

        # Use the largest contour (should be the text line)
        cnt = max(contours, key=cv2.contourArea)
        
        # Get bounding box for visualization
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Polygon coordinates from the actual contour (simplified)
        epsilon = 0.001 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        coords = " ".join([f"{pt[0][0]},{pt[0][1]}" for pt in approx])

        # Baseline = middle horizontal line of bounding box
        baseline_y = y + h // 2
        baseline_points = f"{x},{baseline_y} {x + w},{baseline_y}"

        # Create TextLine with polygon coordinates
        textline = ET.SubElement(region, "TextLine", {"id": f"textline_{i}", "custom": "0"})
        ET.SubElement(textline, "Coords", {"points": coords})
        ET.SubElement(textline, "Baseline", {"points": baseline_points})
        text_equiv = ET.SubElement(textline, "TextEquiv")
        ET.SubElement(text_equiv, "Unicode").text = ""

    # --- save PAGE XML ---
    output_dir = Path("xml_output_bbox")
    output_dir.mkdir(exist_ok=True)
    xml_path = output_dir / f"{file_name}.xml"

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")  # Pretty print
    tree.write(xml_path, encoding="utf-8", xml_declaration=True)
    print(f"Saved PAGE XML → {xml_path}")

# CSV output section - generate bounding box coordinates to CSV
import csv

# Create output directory for CSV
csv_output_dir = Path("csv_output_bbox")
csv_output_dir.mkdir(exist_ok=True)

# Process all images and save bounding boxes to CSV
for img in imgs:
    orig_pic = os.path.join(dt_dir, img)
    mask_dir = "../datasets/CHAMDoc/GT_line_by_line/"
    file_name = os.path.basename(orig_pic)[:-4]

    # --- find matching masks ---
    matches = []
    for root_dir, dirnames, filenames in os.walk(mask_dir):
        for filename in fnmatch.filter(filenames, f"*{file_name}*"):
            matches.append(os.path.join(root_dir, filename))

    # --- natural sorting ---
    matches = sorted(matches, key=lambda x: [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', x)])

    # --- load original ---
    orig = cv2.imread(orig_pic)
    if orig is None:
        continue
    height, width = orig.shape[:2]

    # Prepare CSV data
    csv_data = []

    # --- process masks and collect bounding boxes ---
    for i, mask_path in enumerate(matches):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        # Ensure binary image
        _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # Apply morphological operations to connect text in lines
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
        dilated = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close, iterations=2)

        # Find contours of text lines
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            continue

        # Use the largest contour (should be the text line)
        cnt = max(contours, key=cv2.contourArea)
        
        # Get bounding box coordinates
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        
        # Store bounding box data
        csv_data.append({
            "image_filename": os.path.basename(orig_pic),
            "textline_id": f"textline_{i}",
            "x": x,
            "y": y,
            "width": w,
            "height": h,
            "area": area
        })

    # --- save to CSV ---
    if csv_data:
        csv_path = csv_output_dir / f"{file_name}.csv"
        fieldnames = ["image_filename", "textline_id", "x", "y", "width", "height", "area"]
        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_data)
        print(f"Saved CSV → {csv_path}")
    else:
        print(f"No text lines found for {file_name}, skipping CSV output")
