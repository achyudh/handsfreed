"""Abstract base class for audio segmentation strategies."""

import asyncio
import logging

import numpy as np

from ..pipeline import AbstractPipelineConsumerComponent

logger = logging.getLogger(__name__)


class SegmentationStrategy(AbstractPipelineConsumerComponent):
    """Abstract base class for audio segmentation strategies."""

    def __init__(
        self,
        raw_audio_queue: asyncio.Queue,
        segment_queue: asyncio.Queue,
        stop_event: asyncio.Event,
        config,
    ):
        """Initialize segmentation strategy.

        Args:
            raw_audio_queue: Queue receiving raw audio frames from audio capture
            segment_queue: Queue to send detected speech segments (np.ndarray)
            stop_event: Event signaling when to stop processing
            config: Application configuration
        """
        super().__init__(raw_audio_queue, segment_queue, stop_event)
        self.config = config

        # Common state
        self._buffer = np.array([], dtype=np.float32)
        self._enabled: bool = False

    async def set_enabled(self, enabled: bool) -> None:
        """Set the enabled state.

        Args:
            enabled: Whether the strategy should be processing audio.
        """
        if enabled != self._enabled:
            logger.info(f"Setting segmentation enabled: {enabled}")
            self._enabled = enabled

            if not enabled:
                self._buffer = np.array([], dtype=np.float32)
