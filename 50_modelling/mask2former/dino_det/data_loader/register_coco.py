from pathlib import Path

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances


def register_text_line_coco_dataset(name, json_file, image_root):
    json_path = Path(json_file)
    image_root_path = Path(image_root)
    if not json_path.exists():
        raise FileNotFoundError(f"COCO annotation file not found: {json_path}")
    if not image_root_path.exists():
        raise FileNotFoundError(f"Image root not found: {image_root_path}")

    if name in DatasetCatalog.list():
        DatasetCatalog.pop(name)

    register_coco_instances(name, {}, str(json_path), str(image_root_path))
    MetadataCatalog.get(name).set(evaluator_type="coco")
    return name