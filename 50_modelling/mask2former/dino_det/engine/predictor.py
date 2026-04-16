import torch

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model


class FullImagePredictor:
    def __init__(self, cfg):
        self.cfg = cfg.clone()
        self.model = build_model(self.cfg)
        self.model.eval()
        DetectionCheckpointer(self.model).load(self.cfg.MODEL.WEIGHTS)

    def __call__(self, image_rgb):
        with torch.no_grad():
            inputs = {
                "image": torch.as_tensor(image_rgb.transpose(2, 0, 1)).float(),
                "height": image_rgb.shape[0],
                "width": image_rgb.shape[1],
            }
            return self.model([inputs])[0]