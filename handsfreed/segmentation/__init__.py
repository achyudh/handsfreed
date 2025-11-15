"""Segmentation strategies for audio processing."""

from .fixed import FixedSegmentationStrategy
from .vad import VADSegmentationStrategy

__all__ = ["FixedSegmentationStrategy", "VADSegmentationStrategy"]
