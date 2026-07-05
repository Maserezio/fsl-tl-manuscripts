import os
import xml.etree.ElementTree as ET
import shutil
from glob import glob
from PIL import Image

def polygon_to_bbox(points):
    xs = [float(x) for x, y in points]
    ys = [float(y) for x, y in points]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    return xmin, ymin, xmax, ymax

def parse_textlines(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    textlines = []
    for tl in root.findall('.//TextLine'):
        coords = tl.find('Coords')
        if coords is not None and 'points' in coords.attrib:
            points_str = coords.attrib['points']
            points = [tuple(map(float, p.split(','))) for p in points_str.split(' ')]
            bbox = polygon_to_bbox(points)
            textlines.append(bbox)
    print(f"Parsed {len(textlines)} textlines from {xml_path}")
    return textlines

def normalize_bbox(bbox, img_width, img_height):
    xmin, ymin, xmax, ymax = bbox
    x_center = (xmin + xmax) / 2 / img_width
    y_center = (ymin + ymax) / 2 / img_height
    width = (xmax - xmin) / img_width
    height = (ymax - ymin) / img_height
    return x_center, y_center, width, height

def process_subset(base_dir, output_dir, subset):
    split_map = {'training': 'train', 'validation': 'val', 'public-test': 'test'}
    ann_base = os.path.join(base_dir, subset, f'PAGE-gt-{subset}', 'PAGE-gt')
    img_base = os.path.join(base_dir, subset, f'img-{subset}', 'img')
    for ann_split, yolo_split in split_map.items():
        ann_dir = os.path.join(ann_base, ann_split)
        yolo_img_dir = os.path.join(output_dir, subset, 'images', yolo_split)
        yolo_lbl_dir = os.path.join(output_dir, subset, 'labels', yolo_split)
        os.makedirs(yolo_img_dir, exist_ok=True)
        os.makedirs(yolo_lbl_dir, exist_ok=True)
        xml_files = glob(os.path.join(ann_dir, '*.xml'))
        print(f"Found {len(xml_files)} XML files in {ann_dir}")
        for xml_file in xml_files:
            textlines = parse_textlines(xml_file)
            img_name = os.path.splitext(os.path.basename(xml_file))[0] + '.jpg'
            img_path = os.path.join(img_base, img_name)
            print(f"Looking for image: {img_path}")
            if os.path.exists(img_path):
                shutil.copy(img_path, yolo_img_dir)
                with Image.open(img_path) as img:
                    img_width, img_height = img.size
                yolo_label_path = os.path.join(yolo_lbl_dir, os.path.splitext(img_name)[0] + '.txt')
                with open(yolo_label_path, 'w') as f:
                    for bbox in textlines:
                        x_center, y_center, width, height = normalize_bbox(bbox, img_width, img_height)
                        f.write(f'0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n')
                print(f"Wrote {len(textlines)} bboxes to {yolo_label_path}")
            else:
                print(f'Image not found: {img_path}')

def main():
    base_dir = '../00_data/DIVA-HisDB'  # Adjust if needed
    output_dir = '../00_data/yolo_dataset_diva_hisdb'
    subsets = ['CB55']
    for subset in subsets:
        print(f'Processing {subset}...')
        process_subset(base_dir, output_dir, subset)
    print('Done.')

if __name__ == '__main__':
    main()
