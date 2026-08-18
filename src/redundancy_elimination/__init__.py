"""Parameter-free adaptation by feature-channel redundancy elimination."""

from .channels import apply_replacements, generate_replacement_pairs
from .models import PointPrompt, ReplacementPair, SearchResult

__all__ = [
    "PointPrompt",
    "ReplacementPair",
    "SearchResult",
    "apply_replacements",
    "generate_replacement_pairs",
]

__version__ = "0.1.0"
