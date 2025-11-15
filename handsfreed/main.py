"""Main entry point for handsfreed daemon."""

import asyncio
import logging
import signal
import sys
from typing import NoReturn

from .config import load_config
from .ipc_server import IPCServer
from .logging_setup import setup_logging
from .pipeline_manager import PipelineManager
from .state import DaemonStateManager

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

    # Create pipeline manager
    try:
        pipeline_manager = PipelineManager(config, stop_event)
        await pipeline_manager.start()
    except Exception as e:
        logger.exception(f"Failed to start pipeline manager: {e}")
        return 1

    # Create IPC server
    ipc_server = IPCServer(
        config.daemon.computed_socket_path,
        state_manager,
        shutdown_event,
        pipeline_manager,
    )

    try:
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
        stop_event.set()  # Signal components to stop
        await pipeline_manager.stop()

        return 0

    except Exception:
        logger.exception("Fatal error in daemon:")
        # Try to clean up if we can
        stop_event.set()
        await pipeline_manager.stop()
        if "ipc_server" in locals():
            await ipc_server.stop()

        return 1

    finally:
        logger.info("Daemon shutdown complete")


def run() -> NoReturn:
    """Entry point for the daemon."""
    asyncio.run(main())
