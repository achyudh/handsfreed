"""Segmentation strategies for audio processing."""

import asyncio
import logging
import numpy as np

from .audio_capture import AUDIO_DTYPE, SAMPLE_RATE
from .pipelines import SegmentationStrategy, TranscriptionTask

logger = logging.getLogger(__name__)


class TimeBasedSegmentationStrategy(SegmentationStrategy):
    """Segments audio based on fixed time durations."""

    def __init__(
        self,
        raw_audio_queue: asyncio.Queue,
        transcription_queue: asyncio.Queue,
        stop_event: asyncio.Event,
        config,
    ):
        """Initialize time-based segmentation strategy.

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
            f"Initialized time-based segmentation (Chunk Duration: {self.chunk_duration_s}s, "
            f"Chunk Size: {self.chunk_size_frames} frames)"
        )

    async def process(self) -> None:
        """Process raw audio frames into fixed-duration chunks."""
        logger.info("Time-based segmentation processor started")

        # Clear buffer when starting
        self._buffer = np.array([], dtype=AUDIO_DTYPE)

        # Create processing task
        self._processing_task = asyncio.current_task()

        try:
            while not self.stop_event.is_set():
                try:
                    # Try to get and process new data (with timeout to check stop event)
                    try:
                        raw_frame = await asyncio.wait_for(
                            self.raw_audio_queue.get(), timeout=0.5
                        )
                    except asyncio.TimeoutError:
                        continue

                    # Append to buffer
                    self._buffer = np.concatenate((self._buffer, raw_frame))

                    # Process buffer if enough data for a chunk
                    while len(self._buffer) >= self.chunk_size_frames:
                        # Extract chunk
                        chunk = self._buffer[: self.chunk_size_frames].astype(
                            AUDIO_DTYPE
                        )

                        # Remove chunk from buffer (non-overlapping)
                        self._buffer = self._buffer[self.chunk_size_frames :]

                        # Only send for transcription if we have an active output mode
                        if self._active_mode is not None:
                            logger.debug(
                                f"Time-based chunk ready: {len(chunk)} frames "
                                f"({len(chunk) / SAMPLE_RATE:.1f}s)"
                            )
                            # Create and send task
                            task = TranscriptionTask(
                                audio=chunk, output_mode=self._active_mode
                            )
                            await self.transcription_queue.put(task)
                        else:
                            logger.debug("Discarding chunk (no active output mode)")

                    # Mark raw frame as processed
                    self.raw_audio_queue.task_done()

                except asyncio.CancelledError:
                    logger.info("Time-based segmentation processor cancelled")
                    break

                except Exception as e:
                    logger.exception(f"Error in time-based segmentation processor: {e}")
                    # Avoid tight loop on error
                    await asyncio.sleep(0.5)

        finally:
            logger.info("Time-based segmentation processor stopped")
