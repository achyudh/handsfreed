"""Pipeline components for audio processing and transcription."""

import abc
import asyncio
import logging
import numpy as np
from dataclasses import dataclass
from typing import Optional

from .ipc_models import CliOutputMode

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionTask:
    """Data structure for passing audio to the transcriber."""

    audio: np.ndarray
    output_mode: CliOutputMode


class SegmentationStrategy(abc.ABC):
    """Abstract base class for audio segmentation strategies."""

    def __init__(
        self,
        raw_audio_queue: asyncio.Queue,
        transcription_queue: asyncio.Queue,
        stop_event: asyncio.Event,
        config,
    ):
        """Initialize segmentation strategy.

        Args:
            raw_audio_queue: Queue receiving raw audio frames from audio capture
            transcription_queue: Queue to send audio segments to transcriber
            stop_event: Event signaling when to stop processing
            config: Application configuration
        """
        self.raw_audio_queue = raw_audio_queue
        self.transcription_queue = transcription_queue
        self.stop_event = stop_event
        self.config = config

        # Common state
        self._buffer = np.array([], dtype=np.float32)
        self._active_mode: Optional[CliOutputMode] = None
        self._processing_task: Optional[asyncio.Task] = None

    @abc.abstractmethod
    async def process(self) -> None:
        """Start processing audio frames into transcription tasks."""
        pass

    async def set_active_output_mode(self, mode: Optional[CliOutputMode]) -> None:
        """Set the active output mode.

        Args:
            mode: Output mode to use, or None to disable output
        """
        if mode != self._active_mode:
            logger.info(f"Setting output mode to: {mode.value if mode else 'None'}")
            self._active_mode = mode

            # Clear buffer when disabling output to avoid processing stale audio
            if mode is None:
                self._buffer = np.array([], dtype=np.float32)

    async def stop(self) -> None:
        """Stop the processing task."""
        if self._processing_task and not self._processing_task.done():
            logger.info(f"Stopping {self.__class__.__name__}...")
            self._processing_task.cancel()
            try:
                await asyncio.wait_for(
                    asyncio.shield(self._processing_task), timeout=2.0
                )
            except asyncio.TimeoutError:
                logger.warning(f"Timeout waiting for {self.__class__.__name__} to stop")
            except asyncio.CancelledError:
                logger.info(f"{self.__class__.__name__} task was cancelled")
            except Exception as e:
                logger.exception(
                    f"Error waiting for {self.__class__.__name__} task: {e}"
                )
            finally:
                self._processing_task = None
                logger.info(f"{self.__class__.__name__} stopped")
