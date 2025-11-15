"""Segments audio based on voice activity detection."""

import asyncio
import collections
import enum
import logging
import time
from typing import Deque, List

import numpy as np

from ..audio_capture import AUDIO_DTYPE, FRAME_SIZE, SAMPLE_RATE
import abc
from ..pipeline import TranscriptionTask
from .strategy import SegmentationStrategy

logger = logging.getLogger(__name__)


class AbstractVADState(abc.ABC):
    """Abstract base class for VAD states."""

    @abc.abstractmethod
    async def handle_vad_result(
        self,
        strategy: "VADSegmentationStrategy",
        speech_prob: float,
        raw_frame: np.ndarray,
    ) -> "AbstractVADState":
        """Handle the VAD result for the current state."""
        raise NotImplementedError

    def __str__(self) -> str:
        """Return the name of the state."""
        return self.__class__.__name__


class SilentState(AbstractVADState):
    """Represents the state where no speech is detected."""

    async def handle_vad_result(
        self,
        strategy: "VADSegmentationStrategy",
        speech_prob: float,
        raw_frame: np.ndarray,
    ) -> "AbstractVADState":
        """Handle VAD result in SILENT state."""
        if speech_prob >= strategy.vad_config.threshold:
            logger.debug(f"VAD: SILENT -> SPEECH (speech_prob={speech_prob})")
            # Add pre-roll buffer content to current segment
            for pre_frame in strategy._pre_roll_buffer:
                strategy._current_segment.append(pre_frame)
            return SpeechState()
        return self


class SpeechState(AbstractVADState):
    """Represents the state where speech is being detected."""

    async def handle_vad_result(
        self,
        strategy: "VADSegmentationStrategy",
        speech_prob: float,
        raw_frame: np.ndarray,
    ) -> "AbstractVADState":
        """Handle VAD result in SPEECH state."""
        strategy._current_segment.append(raw_frame)

        # Check if max speech duration exceeded
        if strategy.vad_config.max_speech_duration_s > 0:
            current_duration_s = (
                sum(len(f) for f in strategy._current_segment) / SAMPLE_RATE
            )
            if current_duration_s >= strategy.vad_config.max_speech_duration_s:
                logger.debug(
                    f"VAD: Max speech duration reached ({current_duration_s:.1f}s)"
                )
                await strategy._finalize_segment()
                return SilentState()

        is_speech_prob_low = speech_prob <= (
            strategy.vad_config.neg_threshold
            if strategy.vad_config.neg_threshold is not None
            else strategy.vad_config.threshold
        )
        if is_speech_prob_low:
            logger.debug(f"VAD: SPEECH -> ENDING_SPEECH (speech_prob={speech_prob})")
            return EndingSpeechState()

        return self


class EndingSpeechState(AbstractVADState):
    """Represents the state where speech has just ended."""

    def __init__(self):
        """Initialize the state."""
        self._silence_start_time = time.monotonic()

    async def handle_vad_result(
        self,
        strategy: "VADSegmentationStrategy",
        speech_prob: float,
        raw_frame: np.ndarray,
    ) -> "AbstractVADState":
        """Handle VAD result in ENDING_SPEECH state."""
        strategy._current_segment.append(raw_frame)

        if speech_prob >= strategy.vad_config.threshold:
            logger.debug(f"VAD: ENDING_SPEECH -> SPEECH (speech_prob={speech_prob})")
            return SpeechState()

        silence_duration_ms = (time.monotonic() - self._silence_start_time) * 1000
        if silence_duration_ms >= strategy.vad_config.min_silence_duration_ms:
            logger.debug(
                f"VAD: ENDING_SPEECH -> SILENT (silence={silence_duration_ms:.0f}ms)"
            )
            await strategy._finalize_segment()
            return SilentState()

        return self


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
        self._current_vad_state: AbstractVADState = SilentState()
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

    async def _consume_item(self, raw_frame: np.ndarray) -> None:
        """Process a single raw audio frame using VAD for speech detection."""
        # Add frame to pre-roll buffer (regardless of active mode)
        self._pre_roll_buffer.append(raw_frame)

        # Only proceed with VAD if we have an active output mode
        if self._active_mode is None:
            return

        # Run VAD model inference (in a thread to avoid blocking)
        try:
            speech_prob = await asyncio.to_thread(self.vad_model, raw_frame, FRAME_SIZE)
        except Exception as e:
            logger.warning(f"Error in VAD inference: {e}")
            # Treat as silent on error
            speech_prob = 0.0

        # Delegate handling to the current state
        new_state = await self._current_vad_state.handle_vad_result(
            self, speech_prob, raw_frame
        )
        if new_state is not self._current_vad_state:
            logger.debug(
                f"VAD state transition: {self._current_vad_state} -> {new_state}"
            )
            self._current_vad_state = new_state

    async def _on_stop(self) -> None:
        """Hook for cleanup logic when the component stops."""
        logger.info("VAD segmentation processor stopped")
        # Finalize any pending segment before exiting
        if self._current_segment:
            await self._finalize_segment()
        # Clear for memory cleanup
        self._current_segment = []
        self._pre_roll_buffer.clear()
        self._current_vad_state = SilentState()

    async def _finalize_segment(self) -> None:
        """Finalize the current speech segment and send for transcription."""
        # Skip if no active output mode or empty segment
        if self._active_mode is None or not self._current_segment:
            self._current_segment = []
            return

        # Check minimum speech duration
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
                return

        # Concatenate audio frames into a single array
        final_audio = np.concatenate(self._current_segment)

        logger.debug(
            f"VAD: Finalizing segment: {len(final_audio)} frames "
            f"({len(final_audio) / SAMPLE_RATE:.1f}s)"
        )

        # Create and send task
        task = TranscriptionTask(audio=final_audio, output_mode=self._active_mode)
        await self.output_queue.put(task)

        # Reset segment
        self._current_segment = []
