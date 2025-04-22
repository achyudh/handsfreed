"""IPC server implementation using Unix domain sockets."""

import asyncio
import logging
from pathlib import Path
from typing import Optional, Set

from .state import DaemonStateManager

logger = logging.getLogger(__name__)

# Size limit for incoming messages (64KB should be plenty for commands)
MAX_MESSAGE_SIZE = 64 * 1024
MESSAGE_TERMINATOR = b"\n"


class IPCServer:
    """Handles IPC communication over Unix domain socket."""

    def __init__(self, socket_path: Path, state_manager: DaemonStateManager):
        """Initialize the IPC server.

        Args:
            socket_path: Path to the Unix domain socket.
            state_manager: The daemon state manager instance.
        """
        self.socket_path = socket_path
        self.state_manager = state_manager
        self._server: Optional[asyncio.Server] = None
        self._client_tasks: Set[asyncio.Task] = set()

    async def start(self) -> None:
        """Start the IPC server.

        Raises:
            OSError: If socket file exists and can't be removed, or other IO errors.
        """
        if self._server:
            logger.warning("Server already started")
            return

        # Clean up existing socket if needed
        if self.socket_path.exists():
            if self.socket_path.is_socket():
                logger.info(f"Removing existing socket file: {self.socket_path}")
                try:
                    self.socket_path.unlink()
                except OSError as e:
                    logger.error(f"Failed to remove existing socket: {e}")
                    raise
            else:
                logger.error(f"Path exists but is not a socket: {self.socket_path}")
                raise OSError(f"Path exists but is not a socket: {self.socket_path}")

        try:
            # Ensure parent directory exists
            self.socket_path.parent.mkdir(parents=True, exist_ok=True)

            # Start the server
            self._server = await asyncio.start_unix_server(
                self._handle_client,
                path=str(self.socket_path),
            )

            logger.info(f"IPC server listening on {self.socket_path}")

        except Exception as e:
            logger.error(f"Failed to start IPC server: {e}")
            if self.socket_path.exists():
                self.socket_path.unlink(missing_ok=True)
            raise

    async def stop(self) -> None:
        """Stop the IPC server and clean up."""
        if not self._server:
            logger.warning("Server not running")
            return

        logger.info("Stopping IPC server...")

        # Close the server
        self._server.close()
        await self._server.wait_closed()
        self._server = None

        # Cancel any active client connections
        if self._client_tasks:
            logger.info(f"Cancelling {len(self._client_tasks)} client tasks...")
            for task in self._client_tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*self._client_tasks, return_exceptions=True)
            self._client_tasks.clear()

        # Clean up socket file
        logger.debug(f"Removing socket file: {self.socket_path}")
        try:
            self.socket_path.unlink(missing_ok=True)
        except OSError as e:
            logger.error(f"Error removing socket file: {e}")

        logger.info("IPC server stopped")

    async def _handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """Handle a client connection.

        Args:
            reader: StreamReader for the client connection
            writer: StreamWriter for the client connection
        """
        peer = writer.get_extra_info("peername") or "Unknown"
        logger.info(f"Client connected: {peer}")

        # Create task and add to set
        task = asyncio.current_task()
        assert task is not None  # for type checking
        self._client_tasks.add(task)

        try:
            while True:
                try:
                    # Read a line (command should end with newline)
                    data = await asyncio.wait_for(
                        reader.readuntil(MESSAGE_TERMINATOR), timeout=5.0
                    )

                    if not data:  # EOF
                        logger.info(f"Client disconnected (EOF): {peer}")
                        break

                    # Remove terminator
                    message = data.rstrip(MESSAGE_TERMINATOR).decode("utf-8")
                    logger.debug(f"Received from {peer}: {message}")

                    # PLACEHOLDER: Echo the message back
                    # Will be replaced with actual command parsing/handling
                    response = f"Echo: {message}\n".encode("utf-8")
                    writer.write(response)
                    await writer.drain()

                except asyncio.TimeoutError:
                    logger.warning(f"Timeout reading from client {peer}")
                    break
                except asyncio.IncompleteReadError:
                    logger.info(f"Client disconnected (incomplete read): {peer}")
                    break
                except ConnectionError as e:
                    logger.warning(f"Connection error with {peer}: {e}")
                    break
                except Exception as e:
                    logger.exception(f"Error handling client {peer}: {e}")
                    break

        finally:
            # Clean up
            logger.info(f"Closing connection with {peer}")
            if not writer.is_closing():
                writer.close()
                try:
                    await asyncio.wait_for(writer.wait_closed(), timeout=1.0)
                except (asyncio.TimeoutError, Exception) as e:
                    logger.warning(f"Error during connection cleanup: {e}")

            self._client_tasks.remove(task)
            logger.debug(f"Connection closed: {peer}")
