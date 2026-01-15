# Manuscript Processing Utilities

This directory contains extracted and organized utility functions from the preprocessing and modelling notebooks.

## Modules

### `xml_utils.py`
XML parsing utilities for PAGE format manuscripts.
- `parse_xml_polygons(xml_path)` - Parse PAGE XML to extract text line polygons
- `create_page_xml(image_path, width, height)` - Create basic PAGE XML structure

### `evaluation_utils.py`
Evaluation utilities for manuscript text line detection and segmentation.
- `hscp_polygon_eval(gt_xml, pred_xml, page_height, page_width, iou_thresh)` - Evaluate polygon predictions from XML files
- `calculate_iou_polygon(poly_a_coords, poly_b_coords)` - Calculate IoU between polygons using Shapely
- `hscp_pixel_eval(gt_mask, pred_mask, iou_thresh)` - Evaluate connected components in binary masks

## Usage

```python
# Import specific functions
from utils.xml_utils import parse_xml_polygons
from utils.evaluation_utils import hscp_polygon_eval

# Or import entire modules
import utils.xml_utils as xml_utils
import utils.evaluation_utils as eval_utils
```

## Dependencies

The utilities require the following packages:
- opencv-python (cv2)
- numpy
- shapely
- xml.etree.ElementTree (built-in)

## Source

These functions were extracted from:
- `10_img_preprocessing/chamdoc_to_polygons.ipynb`
- `50_modelling/bbox_to_polygon_segm.ipynb`