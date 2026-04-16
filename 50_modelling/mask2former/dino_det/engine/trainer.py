import os

import torch
import torch.nn as nn

from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator

from dino_det.data_loader import TextLineDatasetMapper


class DinoMask2FormerTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=TextLineDatasetMapper(cfg, is_train=True))

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name, mapper=TextLineDatasetMapper(cfg, is_train=False))

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        output_folder = output_folder or os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, tasks=("segm",), output_dir=output_folder)

    @classmethod
    def build_optimizer(cls, cfg, model):
        params = []
        memo = set()
        norm_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)

        for module_name, module in model.named_modules():
            for parameter_name, parameter in module.named_parameters(recurse=False):
                if not parameter.requires_grad or parameter in memo:
                    continue
                memo.add(parameter)
                learning_rate = cfg.SOLVER.BASE_LR
                if module_name.startswith("backbone"):
                    learning_rate = cfg.SOLVER.BASE_LR * cfg.SOLVER.BACKBONE_MULTIPLIER
                weight_decay = cfg.SOLVER.WEIGHT_DECAY_NORM if isinstance(module, norm_types) else cfg.SOLVER.WEIGHT_DECAY
                params.append(
                    {
                        "params": [parameter],
                        "lr": learning_rate,
                        "weight_decay": weight_decay,
                    }
                )

        return torch.optim.AdamW(params, lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)