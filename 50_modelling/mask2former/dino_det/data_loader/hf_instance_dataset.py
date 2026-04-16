import json
from io import BytesIO
from pathlib import Path

from PIL import Image


def _ensure_pil_image(image_value):
    if isinstance(image_value, Image.Image):
        return image_value
    if isinstance(image_value, dict):
        if image_value.get("bytes") is not None:
            return Image.open(BytesIO(image_value["bytes"])).convert("RGB")
        if image_value.get("path"):
            return Image.open(image_value["path"]).convert("RGB")
    raise TypeError(f"Unsupported Hugging Face image payload: {type(image_value)!r}")


def _iter_objects(objects_value):
    if isinstance(objects_value, list):
        for obj in objects_value:
            yield obj
        return

    if isinstance(objects_value, dict):
        keys = list(objects_value.keys())
        if not keys:
            return
        length = len(objects_value[keys[0]])
        for index in range(length):
            yield {key: objects_value[key][index] for key in keys}
        return

    raise TypeError(f"Unsupported Hugging Face objects payload: {type(objects_value)!r}")


def _normalize_segmentation(segmentation_value):
    polygons = []
    for polygon in segmentation_value:
        polygons.append([float(coord) for coord in polygon])
    return polygons


def _export_split_to_coco(dataset_split, output_root, split_name):
    split_root = output_root / split_name
    images_root = split_root / "images"
    images_root.mkdir(parents=True, exist_ok=True)

    objects_feature = dataset_split.features["objects"]
    category_feature = objects_feature["category"].feature if hasattr(objects_feature["category"], "feature") else objects_feature["category"]
    categories = [
        {"id": idx + 1, "name": name, "supercategory": name}
        for idx, name in enumerate(category_feature.names)
    ]

    coco = {
        "info": {"description": f"Exported from Hugging Face split '{split_name}'"},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": categories,
    }

    annotation_id = 1
    for row in dataset_split:
        image = _ensure_pil_image(row["image"])
        image_id = int(row["image_id"])

        image_filename = f"{image_id:06d}.png"
        image_path = images_root / image_filename
        image.save(image_path)

        coco["images"].append(
            {
                "id": image_id,
                "file_name": image_filename,
                "width": int(row["width"]),
                "height": int(row["height"]),
            }
        )

        for obj in _iter_objects(row["objects"]):
            segmentation = _normalize_segmentation(obj["segmentation"])
            coco["annotations"].append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": int(obj["category"]) + 1,
                    "bbox": [float(value) for value in obj["bbox"]],
                    "segmentation": segmentation,
                    "area": float(obj["area"]),
                    "iscrowd": int(obj["iscrowd"]),
                }
            )
            annotation_id += 1

    annotations_path = split_root / "annotations.json"
    with open(annotations_path, "w", encoding="utf-8") as handle:
        json.dump(coco, handle, indent=2, ensure_ascii=True)

    return annotations_path, images_root


def prepare_hf_instance_dataset(dataset_name, output_root, train_split="train", val_split="validation", cache_dir=None):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The 'datasets' package is required for --hf-dataset. Install it with 'pip install datasets'."
        ) from exc

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(dataset_name, cache_dir=cache_dir)
    if train_split not in dataset:
        raise ValueError(f"Train split '{train_split}' not found in dataset {dataset_name!r}")
    if val_split not in dataset:
        raise ValueError(f"Validation split '{val_split}' not found in dataset {dataset_name!r}")

    train_json, train_images = _export_split_to_coco(dataset[train_split], output_root, train_split)
    val_json, val_images = _export_split_to_coco(dataset[val_split], output_root, val_split)
    return {
        "train_json": str(train_json),
        "train_images": str(train_images),
        "val_json": str(val_json),
        "val_images": str(val_images),
    }