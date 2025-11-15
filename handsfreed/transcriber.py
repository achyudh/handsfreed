"""Audio transcription module using faster-whisper."""

import asyncio
import logging
from typing import Optional, Tuple

import numpy as np
from faster_whisper import WhisperModel

from .config import AppConfig
from .pipeline import AbstractPipelineConsumerComponent, TranscriptionTask

logger = logging.getLogger(__name__)


class TranscriptionResult:
    """Result of a transcription with metadata."""

    def __init__(
        self,
        text: str,
        language: Optional[str] = None,
        language_probability: Optional[float] = None,
        duration: Optional[float] = None,
    ):
        self.text = text
        self.language = language
        self.language_probability = language_probability
        self.duration = duration

    def __str__(self) -> str:
        return self.text


class Transcriber(AbstractPipelineConsumerComponent):
    """Handles audio transcription using faster-whisper."""

    def __init__(
        self,
        config: AppConfig,
        transcription_queue: asyncio.Queue,
        output_queue: asyncio.Queue,
        stop_event: asyncio.Event,
    ):
        """Initialize the transcriber.

        Args:
            config: Application configuration.
            transcription_queue: Queue to receive TranscriptionTask objects.
            output_queue: Queue to put (text, mode) tuples onto.
            stop_event: Event to signal when to stop processing.
        """
        super().__init__(transcription_queue, output_queue, stop_event)
        self.whisper_config = config.whisper
        self._model: Optional[WhisperModel] = None

    def load_model(self) -> bool:
        """Load the Whisper model.

        Returns:
            True if model loaded successfully, False otherwise.
        """
        if self._model:
            logger.warning("Model already loaded")
            return True

        try:
            logger.info(
                f"Loading Whisper model '{self.whisper_config.model}' "
                f"(Device: {self.whisper_config.device}, "
                f"Compute: {self.whisper_config.compute_type}, "
                f"CPU threads: {self.whisper_config.cpu_threads})"
            )
            self._model = WhisperModel(
                self.whisper_config.model,
                device=self.whisper_config.device,
                compute_type=self.whisper_config.compute_type,
                download_root=None,  # Use default location
                cpu_threads=self.whisper_config.cpu_threads,
            )
            logger.info("Whisper model loaded successfully")
            return True

        except Exception as e:
            logger.exception(f"Failed to load Whisper model: {e}")
            self._model = None
            return False

    def _run_transcription(
        self, audio_chunk: np.ndarray
    ) -> Tuple[Optional[TranscriptionResult], Optional[str]]:
        """Run transcription in a thread.

        Args:
            audio_chunk: Audio data as numpy array.

        Returns:
            Tuple of (TranscriptionResult or None, error message or None).
        """
        if not self._model:
            return None, "Model not loaded"

        try:
            # Run transcription
            segments_generator, info = self._model.transcribe(
                audio_chunk,
                language=self.whisper_config.language,
                beam_size=self.whisper_config.beam_size,
                vad_filter=False,  # Prefer VAD segmentation strategy over model VAD
            )

            # Process segments into full text
            segments = list(segments_generator)
            if not segments:
                return None, None  # No speech detected

            full_text = " ".join(seg.text for seg in segments).strip()
            if not full_text:
                return None, None  # Empty result

            # Return result with metadata
            return (
                TranscriptionResult(
                    text=full_text,
                    language=info.language,
                    language_probability=info.language_probability,
                    duration=info.duration,
                ),
                None,
            )

        except Exception as e:
            logger.exception("Error during transcription")
            return None, str(e)

    async def _consume_item(self, task: TranscriptionTask) -> None:
        """Process a single transcription task."""
        if not self._model:
            logger.error("Cannot process item: Model not loaded")
            return

        # Run transcription in thread pool
        result, error = await asyncio.to_thread(self._run_transcription, task.audio)

        if error:
            logger.error(f"Transcription error: {error}")
            return

        if result and result.text:
            logger.info(
                f"Transcribed [{result.language or 'unknown'}] "
                f"for {task.output_mode.value}: {result.text[:100]}..."
            )
            if result.language_probability is not None:
                logger.debug(f"Language probability: {result.language_probability:.2f}")

            # Put transcription and output mode on output queue
            if self.output_queue:
                await self.output_queue.put((result.text, task.output_mode))
        else:
            logger.debug("No transcription result")

    async def _on_start(self) -> None:
        """Hook for setup logic before the component starts."""
        if not self._model:
            logger.error("Cannot start transcription: Model not loaded")
            raise RuntimeError("Model not loaded")
