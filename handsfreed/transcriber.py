"""Audio transcription module using faster-whisper."""

import asyncio
import logging
from typing import Optional, Tuple

import numpy as np
from faster_whisper import WhisperModel

from .config import AppConfig

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


class Transcriber:
    """Handles audio transcription using faster-whisper."""

    def __init__(
        self,
        config: AppConfig,
        transcription_queue: asyncio.Queue,
        output_queue: asyncio.Queue,
    ):
        """Initialize the transcriber.

        Args:
            config: Application configuration.
            transcription_queue: Queue to receive TranscriptionTask objects.
            output_queue: Queue to put (text, mode) tuples onto.
        """
        self.whisper_config = config.whisper
        self.transcription_queue = transcription_queue
        self.output_queue = output_queue

        # Initialize internals
        self._model: Optional[WhisperModel] = None
        self._transcription_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()

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

    async def _transcription_loop(self) -> None:
        """Main transcription loop."""
        if not self._model:
            logger.error("Cannot start transcription loop: Model not loaded")
            return

        logger.info("Transcription loop started")

        while not self._stop_event.is_set():
            try:
                # Get a transcription task (with timeout to check stop event)
                try:
                    task = await asyncio.wait_for(
                        self.transcription_queue.get(), timeout=0.5
                    )
                except asyncio.TimeoutError:
                    continue

                try:
                    # Run transcription in thread pool
                    result, error = await asyncio.to_thread(
                        self._run_transcription, task.audio
                    )

                    if error:
                        logger.error(f"Transcription error: {error}")
                        continue

                    if result and result.text:
                        logger.info(
                            f"Transcribed [{result.language or 'unknown'}] "
                            f"for {task.output_mode.value}: {result.text[:100]}..."
                        )
                        if result.language_probability is not None:
                            logger.debug(
                                f"Language probability: {result.language_probability:.2f}"
                            )

                        # Put transcription and output mode on output queue
                        await self.output_queue.put((result.text, task.output_mode))
                    else:
                        logger.debug("No transcription result")

                finally:
                    # Mark task as done
                    self.transcription_queue.task_done()

            except asyncio.CancelledError:
                logger.info("Transcription loop cancelled")
                break

            except Exception as e:
                logger.exception(f"Error in transcription loop: {e}")
                # Avoid tight loop on unexpected error
                await asyncio.sleep(0.5)

        logger.info("Transcription loop stopped")

    async def start(self) -> bool:
        """Start the transcription loop.

        Returns:
            True if started successfully, False otherwise.
        """
        if self._transcription_task and not self._transcription_task.done():
            logger.warning("Transcription task already running")
            return True

        if not self._model:
            logger.error("Cannot start transcription: Model not loaded")
            return False

        logger.info("Starting transcription loop")
        self._stop_event.clear()
        self._transcription_task = asyncio.create_task(self._transcription_loop())
        return True

    async def stop(self) -> None:
        """Stop the transcription loop."""
        if not self._transcription_task or self._transcription_task.done():
            logger.warning("Transcription task not running")
            return

        logger.info("Stopping transcription loop")
        self._stop_event.set()

        try:
            # Wait with timeout for task to finish
            await asyncio.wait_for(self._transcription_task, timeout=5.0)
            logger.info("Transcription loop stopped gracefully")
        except asyncio.TimeoutError:
            logger.warning("Timeout waiting for transcription loop to stop")
            self._transcription_task.cancel()
            # Wait a bit more after cancelling
            await asyncio.sleep(0.1)
        except Exception as e:
            logger.exception(f"Error stopping transcription loop: {e}")
        finally:
            self._transcription_task = None
