from datetime import datetime
from pathlib import Path
import xml.etree.ElementTree as ET


PAGE_NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"


def _points_to_string(points):
    return " ".join(f"{int(x)},{int(y)}" for x, y in points)


def write_pagexml(image_path, image_size, text_lines, output_path, creator="mask2former"):
    width, height = image_size
    root = ET.Element(
        "PcGts",
        {
            "xmlns": PAGE_NS,
            "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
            "xsi:schemaLocation": f"{PAGE_NS} {PAGE_NS}/pagecontent.xsd",
        },
    )
    metadata = ET.SubElement(root, "Metadata")
    ET.SubElement(metadata, "Creator").text = creator
    now = datetime.now().isoformat()
    ET.SubElement(metadata, "Created").text = now
    ET.SubElement(metadata, "LastChange").text = now

    page = ET.SubElement(
        root,
        "Page",
        {
            "imageFilename": Path(image_path).name,
            "imageWidth": str(width),
            "imageHeight": str(height),
        },
    )
    region = ET.SubElement(page, "TextRegion", {"id": "region_0001"})
    ET.SubElement(region, "Coords", {"points": f"0,0 {width},0 {width},{height} 0,{height}"})

    for index, text_line in enumerate(text_lines):
        line = ET.SubElement(region, "TextLine", {"id": f"line_{index:04d}"})
        ET.SubElement(line, "Coords", {"points": _points_to_string(text_line["coords"])})
        ET.SubElement(line, "Baseline", {"points": _points_to_string(text_line["baseline"])})

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(output_path, encoding="utf-8", xml_declaration=True)