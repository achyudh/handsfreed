"""Abstract base class for audio segmentation strategies."""

import asyncio
import logging
from typing import Optional

import numpy as np

from ..ipc_models import CliOutputMode
from ..pipeline import AbstractPipelineConsumerComponent

logger = logging.getLogger(__name__)


class SegmentationStrategy(AbstractPipelineConsumerComponent):
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
        super().__init__(raw_audio_queue, transcription_queue, stop_event)
        self.config = config

        # Common state
        self._buffer = np.array([], dtype=np.float32)
        self._active_mode: Optional[CliOutputMode] = None

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
