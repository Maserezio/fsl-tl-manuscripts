import time
import os
import logging
import weakref

import torch
import torch.nn as nn

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.engine import DefaultTrainer
from detectron2.engine.defaults import create_ddp_model
from detectron2.engine.train_loop import TrainerBase
from detectron2.evaluation import COCOEvaluator
from detectron2.engine.train_loop import SimpleTrainer
from detectron2.utils import comm
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import setup_logger

from dino_det.data_loader import TextLineDatasetMapper


class ModernAMPTrainer(SimpleTrainer):
    def __init__(
        self,
        model,
        data_loader,
        optimizer,
        gather_metric_period=1,
        zero_grad_before_forward=False,
        grad_scaler=None,
        precision: torch.dtype = torch.float16,
        log_grad_scaler: bool = False,
        async_write_metrics=False,
    ):
        super().__init__(
            model,
            data_loader,
            optimizer,
            gather_metric_period,
            zero_grad_before_forward,
            async_write_metrics,
        )
        self.grad_scaler = grad_scaler or torch.amp.GradScaler("cuda")
        self.precision = precision
        self.log_grad_scaler = log_grad_scaler

    def run_step(self):
        assert self.model.training, "[ModernAMPTrainer] model was changed to eval mode!"
        assert torch.cuda.is_available(), "[ModernAMPTrainer] CUDA is required for AMP training!"

        start = time.perf_counter()
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        if self.zero_grad_before_forward:
            self.optimizer.zero_grad()

        with torch.amp.autocast(device_type="cuda", dtype=self.precision):
            loss_dict = self.model(data)
            if isinstance(loss_dict, torch.Tensor):
                losses = loss_dict
                loss_dict = {"total_loss": loss_dict}
            else:
                losses = sum(loss_dict.values())

        if not self.zero_grad_before_forward:
            self.optimizer.zero_grad()

        self.grad_scaler.scale(losses).backward()

        if self.log_grad_scaler:
            storage = get_event_storage()
            storage.put_scalar("[metric]grad_scaler", self.grad_scaler.get_scale())

        self.after_backward()

        if self.async_write_metrics:
            self.concurrent_executor.submit(
                self._write_metrics, loss_dict, data_time, iter=self.iter
            )
        else:
            self._write_metrics(loss_dict, data_time)

        self.grad_scaler.step(self.optimizer)
        # Torch's scheduler warning relies on this flag, but GradScaler does not always set it.
        setattr(self.optimizer, "_opt_called", True)
        self.grad_scaler.update()

    def state_dict(self):
        ret = super().state_dict()
        ret["grad_scaler"] = self.grad_scaler.state_dict()
        return ret

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        grad_scaler_state = state_dict.get("grad_scaler")
        if grad_scaler_state is not None:
            self.grad_scaler.load_state_dict(grad_scaler_state)


class DinoMask2FormerTrainer(DefaultTrainer):
    def __init__(self, cfg):
        TrainerBase.__init__(self)
        logger = logging.getLogger("detectron2")
        if not logger.isEnabledFor(logging.INFO):
            setup_logger()

        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())

        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)

        model = create_ddp_model(model, broadcast_buffers=False)
        trainer_cls = ModernAMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer
        self._trainer = trainer_cls(model, data_loader, optimizer)

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        self.checkpointer = DetectionCheckpointer(
            model,
            cfg.OUTPUT_DIR,
            trainer=weakref.proxy(self),
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.register_hooks(self.build_hooks())

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