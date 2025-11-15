"""Pipeline components for audio processing and transcription."""

import abc
import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from .ipc_models import CliOutputMode

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionTask:
    """Data structure for passing audio to the transcriber."""

    audio: np.ndarray
    output_mode: CliOutputMode


class AbstractPipelineComponent(abc.ABC):
    """Abstract base class for all pipeline components."""

    def __init__(self, stop_event: asyncio.Event):
        """Initialize the component."""
        self.stop_event = stop_event
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start the component's processing loop."""
        if self._task and not self._task.done():
            logger.warning(f"{self.__class__.__name__} is already running.")
            return

        logger.info(f"Starting {self.__class__.__name__}...")
        await self._on_start()
        self._task = asyncio.create_task(self._task_loop())

    async def stop(self) -> None:
        """Stop the component's processing loop."""
        if not self._task or self._task.done():
            logger.info(f"{self.__class__.__name__} is not running.")
            return

        logger.info(f"Stopping {self.__class__.__name__}...")
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            logger.info(f"{self.__class__.__name__} task was cancelled.")
        except Exception as e:
            logger.exception(f"Error during {self.__class__.__name__} shutdown: {e}")
        finally:
            await self._on_stop()
            self._task = None
            logger.info(f"{self.__class__.__name__} stopped.")

    @abc.abstractmethod
    async def _task_loop(self) -> None:
        """The main processing/producing loop for the component."""
        raise NotImplementedError

    async def _on_start(self) -> None:
        """Hook for setup logic before the component starts."""
        pass

    async def _on_stop(self) -> None:
        """Hook for cleanup logic when the component stops."""
        pass


class AbstractPipelineConsumerComponent(AbstractPipelineComponent):
    """Abstract base class for pipeline components that consume items from a queue."""

    def __init__(
        self,
        input_queue: asyncio.Queue,
        output_queue: Optional[asyncio.Queue],
        stop_event: asyncio.Event,
    ):
        """Initialize the component."""
        super().__init__(stop_event)
        self.input_queue = input_queue
        self.output_queue = output_queue

    async def _task_loop(self) -> None:
        """The main processing loop for the component."""
        try:
            while not self.stop_event.is_set():
                try:
                    item = await asyncio.wait_for(self.input_queue.get(), timeout=0.5)
                    await self._consume_item(item)
                    self.input_queue.task_done()
                except asyncio.TimeoutError:
                    continue
                except asyncio.CancelledError:
                    logger.info(f"{self.__class__.__name__} processing loop cancelled.")
                    break
                except Exception as e:
                    logger.exception(
                        f"Error in {self.__class__.__name__} processing loop: {e}"
                    )
                    await asyncio.sleep(0.5)  # Avoid tight loop on error
        finally:
            logger.info(f"{self.__class__.__name__} processing loop finished.")

    @abc.abstractmethod
    async def _consume_item(self, item: Any) -> None:
        """Process a single item from the input queue."""
        raise NotImplementedError


class AbstractPipelineProducerComponent(AbstractPipelineComponent):
    """Abstract base class for pipeline components that produce items."""

    def __init__(self, output_queue: asyncio.Queue, stop_event: asyncio.Event):
        """Initialize the component."""
        super().__init__(stop_event)
        self.output_queue = output_queue

    async def _task_loop(self) -> None:
        """The main producing loop for the component."""
        try:
            while not self.stop_event.is_set():
                try:
                    item = await self._produce_item()
                    if item is not None:
                        await self.output_queue.put(item)
                except asyncio.CancelledError:
                    logger.info(f"{self.__class__.__name__} producing loop cancelled.")
                    break
                except Exception as e:
                    logger.exception(
                        f"Error in {self.__class__.__name__} producing loop: {e}"
                    )
                    await asyncio.sleep(0.5)  # Avoid tight loop on error
        finally:
            logger.info(f"{self.__class__.__name__} producing loop finished.")

    @abc.abstractmethod
    async def _produce_item(self) -> Any:
        """Produce a single item to be placed on the output queue."""
        raise NotImplementedError
