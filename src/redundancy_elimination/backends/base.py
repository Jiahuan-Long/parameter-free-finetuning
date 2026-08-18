from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np

from ..models import EncodedImage, PointPrompt, ReplacementPair


class SegmentationBackend(Protocol):
    @property
    def channel_count(self) -> int: ...

    def encode(self, image: np.ndarray) -> EncodedImage: ...

    def predict(
        self,
        encoded: EncodedImage,
        prompt: PointPrompt,
        pairs: Sequence[ReplacementPair],
        *,
        sequential: bool = False,
    ) -> np.ndarray: ...
