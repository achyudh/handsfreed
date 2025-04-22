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
from .transcriber import Transcriber

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

    # Create state manager and shutdown event
    state_manager = DaemonStateManager()
    shutdown_event = asyncio.Event()

    try:
        # Create the processing queues
        audio_queue = asyncio.Queue()  # Audio chunks from capture to transcriber
        output_queue = asyncio.Queue()  # Text + mode from transcriber to output

        # Create the components
        audio_capture = AudioCapture(audio_queue)
        transcriber = Transcriber(config, audio_queue, output_queue)
        output_handler = OutputHandler(config.output)

        # Load the Whisper model (can take a while)
        if not transcriber.load_model():
            logger.error("Failed to load Whisper model")
            return 1

        # Create IPC server with all components
        ipc_server = IPCServer(
            config.daemon.computed_socket_path,
            state_manager,
            shutdown_event,
            transcriber,
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
        await ipc_server.stop()  # This also stops audio/transcription/output

        return 0

    except Exception as e:
        logger.exception("Fatal error in daemon:")
        # Try to clean up if we can
        if "ipc_server" in locals():
            await ipc_server.stop()
        return 1

    finally:
        logger.info("Daemon shutdown complete")


def run() -> NoReturn:
    """Entry point for the daemon."""
    sys.exit(asyncio.run(main()))
