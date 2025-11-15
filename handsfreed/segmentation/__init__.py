"""Segmentation strategies for the audio pipeline."""

__all__ = [
    "FixedSegmentationStrategy",
    "VADSegmentationStrategy",
    "create_segmentation_strategy",
]

import logging

from .fixed import FixedSegmentationStrategy
from .vad import VADSegmentationStrategy

logger = logging.getLogger(__name__)


def create_segmentation_strategy(
    config, raw_audio_queue, transcription_queue, stop_event
):
    """Factory function to create the appropriate segmentation strategy."""
    if not config.vad.enabled:
        logger.info("Using fixed-duration segmentation")
        return FixedSegmentationStrategy(
            raw_audio_queue,
            transcription_queue,
            stop_event,
            config,
        )

    # VAD is enabled, now try to import and load it
    try:
        from faster_whisper.vad import get_vad_model

        logger.info("Loading VAD model...")
        vad_model = get_vad_model()
        logger.info("Using VAD-based segmentation")
        return VADSegmentationStrategy(
            raw_audio_queue,
            transcription_queue,
            stop_event,
            config,
            vad_model,
        )
    except ImportError as e:
        logger.warning(
            f"VAD is enabled, but failed to import VAD module: {e}. "
            "Falling back to fixed-duration segmentation."
        )
    except Exception as e:
        logger.error(f"Failed to load VAD model: {e}")
        logger.info("Falling back to fixed-duration segmentation")

    # Fallback case
    return FixedSegmentationStrategy(
        raw_audio_queue,
        transcription_queue,
        stop_event,
        config,
    )
