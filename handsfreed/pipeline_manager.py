import asyncio
import logging
from typing import Optional

from .audio_capture import AudioCapture
from .config import AppConfig
from .output_handler import OutputHandler
from .segmentation import create_segmentation_strategy
from .transcriber import Transcriber
from .task_assembler import TaskAssembler
from .ipc_models import CliOutputMode
from .state import DaemonStateEnum, DaemonStateManager


logger = logging.getLogger(__name__)


class PipelineManager:
    """Manages the audio processing pipeline."""

    def __init__(
        self,
        config: AppConfig,
        stop_event: asyncio.Event,
        state_manager: DaemonStateManager,
    ):
        """Initialize the pipeline manager."""
        self.config = config
        self.stop_event = stop_event
        self.state_manager = state_manager

        self._auto_disable_event = asyncio.Event()
        self._auto_disable_task: Optional[asyncio.Task] = None

        # Create processing queues
        self.raw_audio_queue = asyncio.Queue()
        self.segment_queue = asyncio.Queue()
        self.transcription_queue = asyncio.Queue()
        self.output_queue = asyncio.Queue()

        # Create component instances
        self.audio_capture = AudioCapture(
            self.raw_audio_queue, self.config.audio, self.stop_event
        )
        self.task_assembler = TaskAssembler(
            self.segment_queue, self.transcription_queue, self.stop_event
        )
        self.transcriber = Transcriber(
            self.config,
            self.transcription_queue,
            self.output_queue,
            self.stop_event,
        )
        self.output_handler = OutputHandler(
            self.config.output, self.output_queue, self.stop_event
        )

        # Create segmentation strategy
        self.segmentation_strategy = create_segmentation_strategy(
            self.config,
            self.raw_audio_queue,
            self.segment_queue,
            self.stop_event,
            self._auto_disable_event,
        )

    async def start(self):
        """Start the pipeline components."""
        # Load the Whisper model
        if not self.transcriber.load_model():
            raise RuntimeError("Failed to load Whisper model")

        # Start pipeline components
        await self.transcriber.start()
        await self.task_assembler.start()
        await self.output_handler.start()
        await self.segmentation_strategy.start()
        await self.audio_capture.start()

        # Start auto-disable monitor task
        self._auto_disable_task = asyncio.create_task(self._monitor_auto_disable())

    async def stop(self):
        """Stop the pipeline components."""
        # Cancel auto-disable monitor task
        if self._auto_disable_task:
            self._auto_disable_task.cancel()
            try:
                await self._auto_disable_task
            except asyncio.CancelledError:
                pass

        await self.audio_capture.stop()
        await self.segmentation_strategy.stop()
        await self.task_assembler.stop()
        await self.transcriber.stop()
        await self.output_handler.stop()

    async def _monitor_auto_disable(self):
        """Monitor for auto-disable events from segmentation strategy."""
        while not self.stop_event.is_set():
            try:
                await self._auto_disable_event.wait()
                if self.stop_event.is_set():
                    break

                logger.info(
                    "Auto-disable triggered by segmentation strategy due to prolonged silence."
                )
                await self.stop_transcription()
                self.state_manager.set_state(DaemonStateEnum.IDLE)
                self._auto_disable_event.clear()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in auto-disable monitor: {e}")
                await asyncio.sleep(1)

    async def start_transcription(self, mode: CliOutputMode):
        """Start the transcription process."""
        await self.audio_capture.start_capture()
        self.task_assembler.set_output_mode(mode)
        await self.segmentation_strategy.set_enabled(True)
        self.output_handler.reset_spacing_state()

    async def stop_transcription(self):
        """Stop the transcription process."""
        await self.segmentation_strategy.set_enabled(False)
        self.task_assembler.set_output_mode(None)
        await self.audio_capture.stop_capture()
        self.output_handler.reset_spacing_state()
