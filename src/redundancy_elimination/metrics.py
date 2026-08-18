from __future__ import annotations

import torch


def binary_iou(prediction: torch.Tensor, target: torch.Tensor) -> float:
    prediction = prediction.bool()
    target = target.bool()
    if prediction.shape != target.shape:
        raise ValueError(
            f"Prediction and target shapes differ: {prediction.shape} != {target.shape}"
        )
    intersection = torch.logical_and(prediction, target).sum(dtype=torch.float64)
    union = torch.logical_or(prediction, target).sum(dtype=torch.float64)
    # Match the convention used by the authors' experimental scripts.
    if union.item() == 0:
        return 0.0
    return float((intersection / union).item())
