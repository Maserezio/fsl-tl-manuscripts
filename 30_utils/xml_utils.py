"""
XML parsing utilities for PAGE format manuscripts
"""
import xml.etree.ElementTree as ET
from typing import List, Tuple


def parse_xml_polygons(xml_path: str) -> List[List[Tuple[int, int]]]:
    """
    Parse PAGE XML file to extract text line polygons.

    Args:
        xml_path: Path to the PAGE XML file

    Returns:
        List of polygons, where each polygon is a list of (x, y) coordinates
    """
    ns = {"ns": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"}
    tree = ET.parse(xml_path)
    root = tree.getroot()

    polygons = []
    for tl in root.findall(".//ns:TextLine", ns):
        coords = tl.find("ns:Coords", ns)
        if coords is None:
            continue
        pts = [(int(x), int(y)) for x, y in
               (p.split(",") for p in coords.attrib["points"].split())]
        polygons.append(pts)
    return polygons


def create_page_xml(image_path: str, width: int, height: int):
    """
    Create a basic PAGE XML structure.

    Args:
        image_path: Path to the image file
        width: Image width
        height: Image height

    Returns:
        Root element and text region element of the XML structure
    """
    from datetime import datetime
    import os

    root = ET.Element("PcGts", {
        "xmlns": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15",
        "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
        "xsi:schemaLocation": (
            "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15 "
            "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15/pagecontent.xsd"
        )
    })

    # Metadata
    metadata = ET.SubElement(root, "Metadata")
    ET.SubElement(metadata, "Creator").text = "Arthur XML Generator"
    ET.SubElement(metadata, "Created").text = datetime.now().isoformat()
    ET.SubElement(metadata, "LastChange").text = datetime.now().isoformat()

    # Page element
    page = ET.SubElement(root, "Page", {
        "imageFilename": os.path.basename(image_path),
        "imageWidth": str(width),
        "imageHeight": str(height)
    })

    # Text region
    region = ET.SubElement(page, "TextRegion", {
        "id": "region_textline",
        "custom": "0"
    })

    ET.SubElement(region, "Coords", {
        "points": f"0,0 {width},0 {width},{height} 0,{height}"
    })

    return root, region
