import asyncio
import logging
from typing import Optional

import numpy as np

from .ipc_models import CliOutputMode
from .pipeline import AbstractPipelineConsumerComponent, TranscriptionTask

logger = logging.getLogger(__name__)


class TaskAssembler(AbstractPipelineConsumerComponent):
    """Assembles TranscriptionTasks from raw audio segments and current context."""

    def __init__(
        self,
        segment_queue: asyncio.Queue,
        transcription_queue: asyncio.Queue,
        stop_event: asyncio.Event,
    ):
        """Initialize the TaskAssembler.

        Args:
            segment_queue: Input queue receiving raw audio segments (np.ndarray).
            transcription_queue: Output queue to send TranscriptionTasks.
            stop_event: Event to signal when to stop processing.
        """
        super().__init__(segment_queue, transcription_queue, stop_event)
        self._active_mode: Optional[CliOutputMode] = None

    def set_output_mode(self, mode: Optional[CliOutputMode]) -> None:
        """Set the active output mode.

        Args:
            mode: Output mode to use for subsequent tasks, or None.
        """
        if mode != self._active_mode:
            logger.info(f"TaskAssembler mode set to: {mode.value if mode else 'None'}")
            self._active_mode = mode

    async def _consume_item(self, audio_segment: np.ndarray) -> None:
        """Process a single audio segment."""
        if self._active_mode is None:
            logger.debug("Dropping audio segment: No active output mode set")
            return

        task = TranscriptionTask(audio=audio_segment, output_mode=self._active_mode)

        if self.output_queue:
            await self.output_queue.put(task)
