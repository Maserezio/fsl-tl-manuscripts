import copy

import cv2
import numpy as np
import torch

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T


class TextLineDatasetMapper:
    def __init__(self, cfg, is_train):
        self.is_train = is_train
        self.image_format = cfg.INPUT.FORMAT
        self.min_size = cfg.INPUT.MIN_SIZE_TRAIN[0] if is_train else cfg.INPUT.MIN_SIZE_TEST
        self.max_size = cfg.INPUT.MAX_SIZE_TRAIN if is_train else cfg.INPUT.MAX_SIZE_TEST
        self.enable_flip = is_train and cfg.INPUT.RANDOM_FLIP == "horizontal"
        crop_size = cfg.INPUT.CROP_SIZE
        self.crop_size = tuple(crop_size) if is_train and crop_size else None

    def _build_transforms(self, image_shape):
        height, width = image_shape[:2]
        transforms = []
        short_edge = min(height, width)
        scale = 1.0
        if short_edge < self.min_size:
            scale = self.min_size / float(short_edge)
            if max(height, width) * scale > self.max_size:
                scale = self.max_size / float(max(height, width))

        if scale != 1.0:
            new_height = int(round(height * scale))
            new_width = int(round(width * scale))
            transforms.append(T.ResizeTransform(height, width, new_height, new_width, interp=cv2.INTER_LINEAR))
            height, width = new_height, new_width

        if self.enable_flip and np.random.rand() < 0.5:
            transforms.append(T.HFlipTransform(width))
        return T.TransformList(transforms)

    def _random_crop(self, image, annotations):
        crop_h, crop_w = self.crop_size
        height, width = image.shape[:2]

        if height <= crop_h and width <= crop_w:
            return image, annotations

        y0 = np.random.randint(0, max(height - crop_h, 0) + 1)
        x0 = np.random.randint(0, max(width - crop_w, 0) + 1)
        y1 = min(y0 + crop_h, height)
        x1 = min(x0 + crop_w, width)

        image = image[y0:y1, x0:x1]

        cropped_annotations = []
        for ann in annotations:
            bx, by, bw, bh = ann["bbox"]
            nx = max(bx - x0, 0)
            ny = max(by - y0, 0)
            nx2 = min(bx + bw - x0, x1 - x0)
            ny2 = min(by + bh - y0, y1 - y0)
            nw = nx2 - nx
            nh = ny2 - ny
            if nw <= 0 or nh <= 0:
                continue
            if nw * nh < 0.2 * bw * bh:
                continue

            new_ann = copy.deepcopy(ann)
            new_ann["bbox"] = [nx, ny, nw, nh]
            if "segmentation" in new_ann and isinstance(new_ann["segmentation"], list):
                new_polys = []
                for poly in new_ann["segmentation"]:
                    shifted = []
                    for i in range(0, len(poly), 2):
                        shifted.append(poly[i] - x0)
                        shifted.append(poly[i + 1] - y0)
                    new_polys.append(shifted)
                new_ann["segmentation"] = new_polys
            cropped_annotations.append(new_ann)

        return image, cropped_annotations

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)
        original_height, original_width = image.shape[:2]

        if not self.is_train:
            transforms = self._build_transforms(image.shape)
            image = transforms.apply_image(image)
            dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1))).float()
            dataset_dict["height"] = original_height
            dataset_dict["width"] = original_width
            dataset_dict.pop("annotations", None)
            return dataset_dict

        annotations = [a for a in dataset_dict.pop("annotations") if a.get("iscrowd", 0) == 0]

        if self.crop_size is not None:
            image, annotations = self._random_crop(image, annotations)

        transforms = self._build_transforms(image.shape)
        image = transforms.apply_image(image)
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1))).float()
        dataset_dict["height"] = image.shape[0]
        dataset_dict["width"] = image.shape[1]

        annotations = [
            utils.transform_instance_annotations(annotation, transforms, image.shape[:2])
            for annotation in annotations
        ]
        instances = utils.annotations_to_instances(annotations, image.shape[:2], mask_format="bitmask")
        dataset_dict["instances"] = utils.filter_empty_instances(instances)
        return dataset_dict