from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from ..channels import apply_replacements
from ..models import EncodedImage, PointPrompt, ReplacementPair


class SAMBackend:
    """Segment Anything (SAM v1) image backend."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        model_type: str = "vit_b",
        device: str = "cuda",
        cache_device: str = "cpu",
    ) -> None:
        try:
            from segment_anything import SamPredictor, sam_model_registry
        except ImportError as exc:
            raise RuntimeError(
                "SAM is not installed. Install this project with the 'sam' extra."
            ) from exc
        model = sam_model_registry[model_type](checkpoint=str(checkpoint))
        model.to(device=device)
        model.eval()
        self.predictor = SamPredictor(model)
        self.device = torch.device(device)
        self.cache_device = torch.device(cache_device)

    @property
    def channel_count(self) -> int:
        return int(self.predictor.model.prompt_encoder.embed_dim)

    @torch.inference_mode()
    def encode(self, image: np.ndarray) -> EncodedImage:
        self.predictor.set_image(image)
        feature = self.predictor.get_image_embedding().detach().to(self.cache_device)
        return EncodedImage(
            feature=feature,
            context={
                "original_size": tuple(self.predictor.original_size),
                "input_size": tuple(self.predictor.input_size),
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
        feature = encoded.feature.to(self.device)
        modified = apply_replacements(
            feature, pairs, channel_dim=1, sequential=sequential
        )
        self.predictor.features = modified
        self.predictor.original_size = tuple(encoded.context["original_size"])
        self.predictor.input_size = tuple(encoded.context["input_size"])
        self.predictor.is_image_set = True
        masks, _, _ = self.predictor.predict(
            point_coords=np.asarray(prompt.points, dtype=np.float32),
            point_labels=np.asarray(prompt.labels, dtype=np.int32),
            multimask_output=False,
        )
        return np.asarray(masks[0], dtype=bool)
