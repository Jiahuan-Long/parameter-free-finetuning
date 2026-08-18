from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from ..channels import apply_replacements
from ..models import EncodedImage, PointPrompt, ReplacementPair


def _move_feature(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().to(device)
    if isinstance(value, list):
        return [_move_feature(item, device) for item in value]
    if isinstance(value, dict):
        return {key: _move_feature(item, device) for key, item in value.items()}
    return deepcopy(value)


class SAM2Backend:
    """SAM 2 image backend using the upstream predictor feature cache."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        model_config: str,
        device: str = "cuda",
        cache_device: str = "cpu",
    ) -> None:
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError as exc:
            message = (
                "SAM 2 is not installed. Follow the upstream installation in "
                "docs/INSTALL.md."
            )
            raise RuntimeError(message) from exc
        model = build_sam2(model_config, str(checkpoint), device=device)
        self.predictor = SAM2ImagePredictor(model)
        self.device = torch.device(device)
        self.cache_device = torch.device(cache_device)

    @property
    def channel_count(self) -> int:
        image_embed = self.predictor._features.get("image_embed")
        if image_embed is None:
            return 256
        return int(image_embed.shape[1])

    @torch.inference_mode()
    def encode(self, image: np.ndarray) -> EncodedImage:
        self.predictor.set_image(image)
        features = _move_feature(self.predictor._features, self.cache_device)
        image_embed = features.pop("image_embed")
        return EncodedImage(
            feature=image_embed,
            context={
                "features": features,
                "orig_hw": deepcopy(self.predictor._orig_hw),
            },
        )

    @torch.inference_mode()
    def predict(
        self,
        encoded: EncodedImage,
        prompt: PointPrompt,
        pairs: Sequence[ReplacementPair],
        *,
        sequential: bool = False,
    ) -> np.ndarray:
        image_embed = apply_replacements(
            encoded.feature.to(self.device),
            pairs,
            channel_dim=1,
            sequential=sequential,
        )
        features = _move_feature(encoded.context["features"], self.device)
        features["image_embed"] = image_embed
        self.predictor._features = features
        self.predictor._orig_hw = deepcopy(encoded.context["orig_hw"])
        self.predictor._is_image_set = True
        masks, _, _ = self.predictor.predict(
            point_coords=np.asarray(prompt.points, dtype=np.float32),
            point_labels=np.asarray(prompt.labels, dtype=np.int32),
            multimask_output=False,
        )
        return np.asarray(masks[0], dtype=bool)
