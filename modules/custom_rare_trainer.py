"""Standalone custom rare-class trainer used in rare-FT stage.
Extracted for delivery; training scripts embed the same code for DDP.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import RANK
from ultralytics.utils.loss import v8DetectionLoss

# Cosmetic=1.5, Nonmetallic_Lighter=3.0 (FT1 selected recipe)
RARE_CLS_WEIGHTS = [1.5, 1.0, 1.0, 3.0, 1.0, 1.0, 1.0, 1.0]
COSMETIC_ID = 0
LIGHTER_ID = 3


class WeightedDetectionLoss(v8DetectionLoss):
    def __init__(self, model, class_weights=None, tal_topk=10, tal_topk2=None):
        super().__init__(model, tal_topk=tal_topk, tal_topk2=tal_topk2)
        if class_weights is not None:
            self.bce = nn.BCEWithLogitsLoss(
                pos_weight=class_weights.to(self.device),
                reduction="none",
            )


class WeightedDetectionModel(DetectionModel):
    def init_criterion(self):
        w = torch.tensor(RARE_CLS_WEIGHTS, dtype=torch.float32)
        return WeightedDetectionLoss(self, class_weights=w)


class RareClassTrainer(DetectionTrainer):
    """Class-weighted cls loss; best.pt by (Cosmetic + Lighter) mAP50 / 2."""

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = WeightedDetectionModel(
            cfg, nc=self.data["nc"], verbose=verbose and RANK in {-1, 0}
        )
        if weights:
            model.load(weights)
        return model

    def validate(self):
        metrics, fitness = super().validate()
        maps50 = None
        validator = getattr(self, "validator", None)
        if validator is not None and getattr(validator, "metrics", None) is not None:
            box = validator.metrics.box
            if hasattr(box, "maps50") and box.maps50 is not None:
                maps50 = box.maps50
            elif hasattr(box, "ap50") and box.ap50 is not None:
                maps50 = box.ap50
        if maps50 is not None and len(maps50) > LIGHTER_ID:
            cos = float(maps50[COSMETIC_ID])
            lig = float(maps50[LIGHTER_ID])
            fitness = (cos + lig) / 2.0
            print(f"Rare-class fitness: {fitness:.5f}  Cosmetic={cos:.5f}  Lighter={lig:.5f}")
        elif isinstance(metrics, dict):
            fitness = float(metrics.get("metrics/mAP50(B)", fitness))
        return metrics, fitness
