import asyncio
import logging

from .audio_capture import AudioCapture
from .config import AppConfig
from .output_handler import OutputHandler
from .segmentation import create_segmentation_strategy
from .transcriber import Transcriber
from .ipc_models import CliOutputMode


logger = logging.getLogger(__name__)


class PipelineManager:
    """Manages the audio processing pipeline."""

    def __init__(self, config: AppConfig, stop_event: asyncio.Event):
        """Initialize the pipeline manager."""
        self.config = config
        self.stop_event = stop_event

        # Create processing queues
        self.raw_audio_queue = asyncio.Queue()
        self.transcription_queue = asyncio.Queue()
        self.output_queue = asyncio.Queue()

        # Create component instances
        self.audio_capture = AudioCapture(
            self.raw_audio_queue, self.config.audio, self.stop_event
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
            self.transcription_queue,
            self.stop_event,
        )

    async def start(self):
        """Start the pipeline components."""
        # Load the Whisper model
        if not self.transcriber.load_model():
            raise RuntimeError("Failed to load Whisper model")

        # Start pipeline components
        await self.transcriber.start()
        await self.output_handler.start()
        await self.segmentation_strategy.start()
        await self.audio_capture.start()

    async def stop(self):
        """Stop the pipeline components."""
        await self.audio_capture.stop()
        await self.segmentation_strategy.stop()
        await self.transcriber.stop()
        await self.output_handler.stop()

    async def start_transcription(self, mode: CliOutputMode):
        """Start the transcription process."""
        await self.segmentation_strategy.set_active_output_mode(mode)
        self.output_handler.reset_spacing_state()

    async def stop_transcription(self):
        """Stop the transcription process."""
        await self.segmentation_strategy.set_active_output_mode(None)
        self.output_handler.reset_spacing_state()
