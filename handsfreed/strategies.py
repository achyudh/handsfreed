"""Segmentation strategies for audio processing."""

import asyncio
import collections
import enum
import logging
import time
from typing import Deque, List

import numpy as np

from .audio_capture import AUDIO_DTYPE, FRAME_SIZE, SAMPLE_RATE
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


class VADState(enum.Enum):
    """States for the VAD state machine."""

    SILENT = "silent"
    SPEECH = "speech"
    ENDING_SPEECH = "ending_speech"


class VADSegmentationStrategy(SegmentationStrategy):
    """Segments audio based on voice activity detection."""

    def __init__(
        self,
        raw_audio_queue: asyncio.Queue,
        transcription_queue: asyncio.Queue,
        stop_event: asyncio.Event,
        config,
        vad_model,
    ):
        """Initialize VAD-based segmentation strategy.

        Args:
            raw_audio_queue: Queue receiving raw audio frames from audio capture
            transcription_queue: Queue to send audio segments to transcriber
            stop_event: Event signaling when to stop processing
            config: Application configuration
            vad_model: Loaded VAD model instance
        """
        super().__init__(raw_audio_queue, transcription_queue, stop_event, config)

        self.vad_config = config.vad
        self.vad_model = vad_model

        # Initialize state
        self._vad_state = VADState.SILENT
        self._silence_start_time = 0
        self._current_segment: List[np.ndarray] = []

        # Calculate pre-roll buffer size in frames
        pre_roll_samples = int(
            self.vad_config.pre_roll_duration_ms * SAMPLE_RATE / 1000
        )
        pre_roll_frames = pre_roll_samples // FRAME_SIZE
        self._pre_roll_buffer: Deque[np.ndarray] = collections.deque(
            maxlen=pre_roll_frames
        )

        logger.info(
            f"Initialized VAD-based segmentation (Threshold: {self.vad_config.threshold}, "
            f"Min Speech: {self.vad_config.min_speech_duration_ms}ms, "
            f"Min Silence: {self.vad_config.min_silence_duration_ms}ms, "
            f"Pre-roll: {self.vad_config.pre_roll_duration_ms}ms, "
            f"Max Speech: {self.vad_config.max_speech_duration_s}s)"
        )

    async def process(self) -> None:
        """Process raw audio frames using VAD for speech detection."""
        logger.info("VAD segmentation processor started")

        # Clear state variables when starting
        self._vad_state = VADState.SILENT
        self._silence_start_time = 0
        self._current_segment = []
        self._pre_roll_buffer.clear()

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

                    # Add frame to pre-roll buffer (regardless of active mode)
                    self._pre_roll_buffer.append(raw_frame)

                    # Only proceed with VAD if we have an active output mode
                    if self._active_mode is None:
                        self.raw_audio_queue.task_done()
                        continue

                    # Run VAD model inference (in a thread to avoid blocking)
                    try:
                        speech_prob = await asyncio.to_thread(
                            self.vad_model, raw_frame, FRAME_SIZE
                        )
                        is_speech_prob_high = speech_prob >= self.vad_config.threshold
                        # Use neg_threshold if defined, otherwise just invert the main threshold check
                        is_speech_prob_low = speech_prob <= (
                            self.vad_config.neg_threshold
                            if self.vad_config.neg_threshold is not None
                            else self.vad_config.threshold
                        )
                    except Exception as e:
                        logger.warning(f"Error in VAD inference: {e}")
                        # Treat as silent on error
                        is_speech_prob_high = False
                        is_speech_prob_low = True

                    # Process audio based on current state
                    if self._vad_state == VADState.SILENT:
                        if is_speech_prob_high:
                            # Transition to SPEECH state
                            logger.debug(
                                f"VAD: SILENT -> SPEECH (speech_prob={speech_prob})"
                            )
                            self._vad_state = VADState.SPEECH

                            # Add pre-roll buffer content to current segment
                            for pre_frame in self._pre_roll_buffer:
                                self._current_segment.append(pre_frame)

                            # Reset the silence timer
                            self._silence_start_time = 0

                    elif self._vad_state == VADState.SPEECH:
                        # Add current frame to segment
                        self._current_segment.append(raw_frame)

                        # Check if max speech duration exceeded
                        if self.vad_config.max_speech_duration_s > 0:
                            current_duration_s = (
                                sum(len(f) for f in self._current_segment) / SAMPLE_RATE
                            )
                            if (
                                current_duration_s
                                >= self.vad_config.max_speech_duration_s
                            ):
                                logger.debug(
                                    f"VAD: Max speech duration reached ({current_duration_s:.1f}s)"
                                )
                                await self._finalize_segment()
                                self._vad_state = VADState.SILENT
                                continue

                        if is_speech_prob_low:
                            # Start timing silence
                            logger.debug(
                                f"VAD: SPEECH -> ENDING_SPEECH (speech_prob={speech_prob})"
                            )
                            self._vad_state = VADState.ENDING_SPEECH
                            self._silence_start_time = time.monotonic()

                    elif self._vad_state == VADState.ENDING_SPEECH:
                        # Always add frame to current segment in ENDING_SPEECH state
                        self._current_segment.append(raw_frame)

                        if is_speech_prob_high:
                            # Resume speech, go back to SPEECH state
                            logger.debug(
                                f"VAD: ENDING_SPEECH -> SPEECH (speech_prob={speech_prob})"
                            )
                            self._vad_state = VADState.SPEECH
                            self._silence_start_time = 0
                        else:
                            # Check if silence duration exceeded threshold
                            silence_duration_ms = (
                                time.monotonic() - self._silence_start_time
                            ) * 1000
                            if (
                                silence_duration_ms
                                >= self.vad_config.min_silence_duration_ms
                            ):
                                logger.debug(
                                    f"VAD: ENDING_SPEECH -> SILENT (silence={silence_duration_ms:.0f}ms)"
                                )
                                await self._finalize_segment()
                                self._vad_state = VADState.SILENT

                    # Mark raw frame as processed
                    self.raw_audio_queue.task_done()

                except asyncio.CancelledError:
                    logger.info("VAD segmentation processor cancelled")
                    # Finalize any pending segment before exiting
                    if self._current_segment:
                        await self._finalize_segment()
                    break

                except Exception as e:
                    logger.exception(f"Error in VAD segmentation processor: {e}")
                    # Avoid tight loop on error
                    await asyncio.sleep(0.5)

        finally:
            # Clear for memory cleanup
            self._current_segment = []
            self._pre_roll_buffer.clear()
            logger.info("VAD segmentation processor stopped")

    async def _finalize_segment(self) -> None:
        """Finalize the current speech segment and send for transcription."""
        # Skip if no active output mode or empty segment
        if self._active_mode is None or not self._current_segment:
            self._current_segment = []
            self._silence_start_time = 0
            return

        # Check minimum speech duration
        logging.debug(f"Current Segment: {len(self._current_segment)}")
        logging.debug(f"Pre Roll Buffer: {len(self._pre_roll_buffer)}")
        if self.vad_config.min_speech_duration_ms > 0:
            speech_duration_ms = (
                sum(len(f) for f in self._current_segment) / SAMPLE_RATE * 1000
            )
            if speech_duration_ms < self.vad_config.min_speech_duration_ms:
                logger.debug(
                    f"VAD: Discarding short segment ({speech_duration_ms:.0f}ms < "
                    f"{self.vad_config.min_speech_duration_ms}ms)"
                )
                self._current_segment = []
                self._silence_start_time = 0
                return

        # Concatenate audio frames into a single array
        final_audio = np.concatenate(self._current_segment)

        logger.debug(
            f"VAD: Finalizing segment: {len(final_audio)} frames "
            f"({len(final_audio) / SAMPLE_RATE:.1f}s)"
        )

        # Create and send task
        task = TranscriptionTask(audio=final_audio, output_mode=self._active_mode)
        await self.transcription_queue.put(task)

        # Reset segment and silence timer
        self._current_segment = []
        self._silence_start_time = 0
