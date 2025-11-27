"""Segments audio based on fixed time durations."""

import asyncio
import logging

import numpy as np

from ..audio_capture import AUDIO_DTYPE, SAMPLE_RATE
from .strategy import SegmentationStrategy

logger = logging.getLogger(__name__)


class FixedSegmentationStrategy(SegmentationStrategy):
    """Segments audio based on fixed time durations."""

    def __init__(
        self,
        raw_audio_queue: asyncio.Queue,
        segment_queue: asyncio.Queue,
        stop_event: asyncio.Event,
        config,
    ):
        """Initialize fixed-duration segmentation strategy.

        Args:
            raw_audio_queue: Queue receiving raw audio frames from audio capture
            segment_queue: Queue to send audio segments to task assembler
            stop_event: Event signaling when to stop processing
            config: Application configuration
        """
        super().__init__(raw_audio_queue, segment_queue, stop_event, config)

        # Get chunk duration from config
        self.chunk_duration_s = config.daemon.time_chunk_s

        # Calculate chunk size in frames
        self.chunk_size_frames = int(self.chunk_duration_s * SAMPLE_RATE)

        logger.info(
            f"Initialized fixed-duration segmentation (Chunk Duration: {self.chunk_duration_s}s, "
            f"Chunk Size: {self.chunk_size_frames} frames)"
        )

    async def _consume_item(self, raw_frame: np.ndarray) -> None:
        """Process a single raw audio frame into fixed-duration chunks."""
        self._buffer = np.concatenate((self._buffer, raw_frame))

        while len(self._buffer) >= self.chunk_size_frames:
            chunk = self._buffer[: self.chunk_size_frames].astype(AUDIO_DTYPE)

            self._buffer = self._buffer[self.chunk_size_frames :]

            if self._enabled:
                logger.debug(
                    f"Fixed-duration chunk ready: {len(chunk)} frames "
                    f"({len(chunk) / SAMPLE_RATE:.1f}s)"
                )
                await self.output_queue.put(chunk)
            else:
                logger.debug("Discarding chunk (segmentation disabled)")

    async def _on_stop(self) -> None:
        """Hook for cleanup logic when the component stops."""
        logger.info("Fixed-duration segmentation processor stopped")
        self._buffer = np.array([], dtype=AUDIO_DTYPE)
