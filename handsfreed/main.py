"""Main entry point for handsfreed daemon."""

import asyncio
import logging
import signal
import sys
from typing import NoReturn

from .audio_capture import AudioCapture
from .config import load_config
from .ipc_server import IPCServer
from .logging_setup import setup_logging
from .output_handler import OutputHandler
from .state import DaemonStateManager
from .strategies import TimeBasedSegmentationStrategy, VADSegmentationStrategy
from .transcriber import Transcriber

# Import VAD model loader
try:
    from faster_whisper.vad import get_vad_model
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"Failed to import VAD model: {e}")
    get_vad_model = None

logger = logging.getLogger(__name__)

__all__ = ["run"]  # Export the run function


async def main() -> int:
    """Main daemon function.

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    # Load configuration first (might need it for logging setup)
    try:
        config = load_config()
    except Exception as e:
        print(f"Error loading configuration: {e}", file=sys.stderr)
        return 1

    # Setup logging
    setup_logging(config.daemon.log_level, config.daemon.computed_log_file)
    logger.info("Starting handsfreed daemon...")

    # Create state manager
    state_manager = DaemonStateManager()

    # Create stop/shutdown events
    stop_event = asyncio.Event()
    shutdown_event = asyncio.Event()

    # Create the processing queues
    raw_audio_queue = asyncio.Queue()  # Raw audio frames from capture to strategy
    transcription_queue = asyncio.Queue()  # Audio chunks from strategy to transcriber
    output_queue = asyncio.Queue()  # Text + mode from transcriber to output

    # Create component instances (but don't start them yet)
    audio_capture = AudioCapture(raw_audio_queue)
    transcriber = Transcriber(config, transcription_queue, output_queue)
    output_handler = OutputHandler(config.output)

    # Create and start tasks
    tasks = []

    try:
        # Load the Whisper model (can take a while)
        if not transcriber.load_model():
            logger.error("Failed to load Whisper model")
            return 1

        # Create segmentation strategy based on configuration
        if config.vad.enabled and get_vad_model is not None:
            try:
                logger.info("Loading VAD model...")
                vad_model = get_vad_model()
                logger.info("Using VAD-based segmentation")
                segmentation_strategy = VADSegmentationStrategy(
                    raw_audio_queue, transcription_queue, stop_event, config, vad_model
                )
            except Exception as e:
                logger.error(f"Failed to load VAD model: {e}")
                logger.info("Falling back to time-based segmentation")
                segmentation_strategy = TimeBasedSegmentationStrategy(
                    raw_audio_queue, transcription_queue, stop_event, config
                )
        else:
            if config.vad.enabled and get_vad_model is None:
                logger.warning(
                    "VAD is enabled in config, but VAD module could not be imported. "
                    "Falling back to time-based segmentation."
                )
            else:
                logger.info("Using time-based segmentation")

            segmentation_strategy = TimeBasedSegmentationStrategy(
                raw_audio_queue, transcription_queue, stop_event, config
            )

        # Start transcriber
        if not await transcriber.start():
            logger.error("Failed to start transcriber")
            return 1

        # Start output handler
        await output_handler.start(output_queue)

        # Create and start segmentation strategy processing task
        strategy_task = asyncio.create_task(segmentation_strategy.process())
        tasks.append(strategy_task)

        # Create IPC server with all components
        ipc_server = IPCServer(
            config.daemon.computed_socket_path,
            state_manager,
            shutdown_event,
            segmentation_strategy,
            audio_capture,
            output_handler,
        )

        # Setup signal handlers
        def handle_signal(sig: int) -> None:
            sig_name = signal.Signals(sig).name
            logger.info(f"Received signal {sig_name}, initiating shutdown...")
            shutdown_event.set()

        # Register signal handlers
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda s=sig: handle_signal(s))

        # Start IPC server to handle user commands
        await ipc_server.start()

        logger.info("Daemon started successfully")

        # Wait for shutdown signal
        await shutdown_event.wait()

        logger.info("Starting graceful shutdown...")

        # Stop in reverse order to avoid dangling tasks
        await ipc_server.stop()
        stop_event.set()  # Signal strategies to stop

        # Cancel and await all tasks
        for task in tasks:
            if not task.done():
                task.cancel()

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        # Stop components
        await segmentation_strategy.stop()
        await transcriber.stop()
        await output_handler.stop()

        return 0

    except Exception:
        logger.exception("Fatal error in daemon:")
        # Try to clean up if we can
        stop_event.set()
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        # Stop components
        if "segmentation_strategy" in locals():
            await segmentation_strategy.stop()
        if "transcriber" in locals():
            await transcriber.stop()
        if "output_handler" in locals():
            await output_handler.stop()
        if "ipc_server" in locals():
            await ipc_server.stop()

        return 1

    finally:
        logger.info("Daemon shutdown complete")


def run() -> NoReturn:
    """Entry point for the daemon."""
    sys.exit(asyncio.run(main()))
