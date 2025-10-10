"""Main entry point for handsfreed daemon."""

import asyncio
import logging
import signal
import sys
from typing import NoReturn

from .audio_capture import PWRecordCapture
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
    # Load configuration first
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

    # Create the processing queues
    transcription_queue = asyncio.Queue()  # Audio chunks from capture to transcriber
    output_queue = asyncio.Queue()  # Text + mode from transcriber to output

    # Create component instances
    audio_capture = PWRecordCapture(config, transcription_queue)
    transcriber = Transcriber(config, transcription_queue, output_queue)
    output_handler = OutputHandler(config.output)

    try:
        # Load the Whisper model (can take a while)
        if not transcriber.load_model():
            logger.error("Failed to load Whisper model")
            return 1

        # Start transcriber and output handler
        if not await transcriber.start():
            logger.error("Failed to start transcriber")
            return 1
        await output_handler.start(output_queue)

        # Create IPC server with all components
        ipc_server = IPCServer(
            config.daemon.computed_socket_path,
            state_manager,
            shutdown_event,
            audio_capture,
            output_handler,
        )

        # Setup signal handlers
        def handle_signal(sig: int) -> None:
            sig_name = signal.Signals(sig).name
            logger.info(f"Received signal {sig_name}, initiating shutdown...")
            shutdown_event.set()

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda s=sig: handle_signal(s))

        # Start IPC server to handle user commands
        await ipc_server.start()

        logger.info("Daemon started successfully")

        # Wait for shutdown signal
        await shutdown_event.wait()

        logger.info("Starting graceful shutdown...")

    except Exception:
        logger.exception("Fatal error in daemon startup:")
        return 1

    finally:
        # Stop in reverse order
        if "ipc_server" in locals() and ipc_server._server:
            await ipc_server.stop()
        if "transcriber" in locals() and transcriber._transcription_task:
            await transcriber.stop()
        if "output_handler" in locals() and output_handler._task:
            await output_handler.stop()

        logger.info("Daemon shutdown complete")

    return 0


def run() -> NoReturn:
    """Entry point for the daemon."""
    try:
        asyncio.run(main())
        sys.exit(0)
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as e:
        # Fallback logger in case of early failure
        logging.basicConfig()
        logger.exception(f"Daemon failed with unhandled exception: {e}")
        sys.exit(1)
