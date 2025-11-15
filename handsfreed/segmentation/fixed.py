"""Segments audio based on fixed time durations."""

import asyncio
import logging

import numpy as np

from ..audio_capture import AUDIO_DTYPE, SAMPLE_RATE
from ..pipeline import TranscriptionTask
from .strategy import SegmentationStrategy

logger = logging.getLogger(__name__)


class FixedSegmentationStrategy(SegmentationStrategy):
    """Segments audio based on fixed time durations."""

    def __init__(
        self,
        raw_audio_queue: asyncio.Queue,
        transcription_queue: asyncio.Queue,
        stop_event: asyncio.Event,
        config,
    ):
        """Initialize fixed-duration segmentation strategy.

        Args:
            raw_audio_queue: Queue receiving raw audio frames from audio capture
            transcription_queue: Queue to send audio segments to transcriber
            stop_event: Event signaling when to stop processing
            config: Application configuration
        """
        super().__init__(raw_audio_queue, transcription_queue, stop_event, config)

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
        # Append to buffer
        self._buffer = np.concatenate((self._buffer, raw_frame))

        # Process buffer if enough data for a chunk
        while len(self._buffer) >= self.chunk_size_frames:
            # Extract chunk
            chunk = self._buffer[: self.chunk_size_frames].astype(AUDIO_DTYPE)

            # Remove chunk from buffer (non-overlapping)
            self._buffer = self._buffer[self.chunk_size_frames :]

            # Only send for transcription if we have an active output mode
            if self._active_mode is not None:
                logger.debug(
                    f"Fixed-duration chunk ready: {len(chunk)} frames "
                    f"({len(chunk) / SAMPLE_RATE:.1f}s)"
                )
                # Create and send task
                task = TranscriptionTask(audio=chunk, output_mode=self._active_mode)
                await self.output_queue.put(task)
            else:
                logger.debug("Discarding chunk (no active output mode)")

    async def _on_stop(self) -> None:
        """Hook for cleanup logic when the component stops."""
        logger.info("Fixed-duration segmentation processor stopped")
        self._buffer = np.array([], dtype=AUDIO_DTYPE)
