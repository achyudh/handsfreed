"""IPC server implementation using Unix domain sockets."""

import asyncio
import json
import logging
from pathlib import Path
from typing import Optional, Set

from pydantic import ValidationError

from .ipc_models import (
    AckResponse,
    CliOutputMode,
    CommandWrapper,
    DaemonStateModel,
    ErrorResponse,
    ResponseWrapper,
    ShutdownCommand,
    StartCommand,
    StateNotification,
    StatusCommand,
    StatusResponse,
    StopCommand,
    SubscribeCommand,
    ToggleCommand,
)
from .pipeline_manager import PipelineManager
from .state import DaemonStateEnum, DaemonStateManager

logger = logging.getLogger(__name__)

# Size limit for incoming messages (64KB should be plenty for commands)
MAX_MESSAGE_SIZE = 64 * 1024
MESSAGE_TERMINATOR = b"\n"


class IPCServer:
    """Handles IPC communication over Unix domain socket."""

    def __init__(
        self,
        socket_path: Path,
        state_manager: DaemonStateManager,
        shutdown_event: asyncio.Event,
        pipeline_manager: PipelineManager,
    ):
        """Initialize the IPC server.

        Args:
            socket_path: Path to the Unix domain socket
            state_manager: Daemon state manager instance
            shutdown_event: Event to signal daemon shutdown
            pipeline_manager: The pipeline manager instance.
        """
        self.socket_path = socket_path
        self.state_manager = state_manager
        self.shutdown_event = shutdown_event
        self.pipeline_manager = pipeline_manager

        self._server: Optional[asyncio.Server] = None
        self._client_tasks: Set[asyncio.Task] = set()
        self._subscribers: Set[asyncio.StreamWriter] = set()

        # Register for state updates
        self.state_manager.add_observer(self._on_state_change)

    def _on_state_change(
        self, new_state: DaemonStateEnum, error: Optional[str]
    ) -> None:
        """Handle state change notifications."""
        if not self._subscribers:
            return

        try:
            state_model = DaemonStateModel(state=new_state, last_error=error)
            notification = ResponseWrapper(root=StateNotification(status=state_model))

            # Create tasks for broadcasting to avoid blocking
            asyncio.create_task(self._broadcast_notification(notification))
        except Exception as e:
            logger.error(f"Error preparing state notification: {e}")

    async def _broadcast_notification(self, notification: ResponseWrapper) -> None:
        """Broadcast a notification to all subscribers."""
        if not self._subscribers:
            return

        data = notification.model_dump_json().encode("utf-8") + MESSAGE_TERMINATOR

        # Copy set to avoid modification during iteration
        subscribers = list(self._subscribers)

        for writer in subscribers:
            if writer.is_closing():
                self._subscribers.discard(writer)
                continue

            try:
                writer.write(data)
                await writer.drain()
            except Exception as e:
                logger.warning(f"Error broadcasting to subscriber: {e}")
                self._subscribers.discard(writer)

    async def _send_response(
        self, writer: asyncio.StreamWriter, response: ResponseWrapper
    ) -> None:
        """Send a response to a client.

        Args:
            writer: StreamWriter to send through
            response: Response to send
        """
        try:
            response_json = response.model_dump_json()
            writer.write(response_json.encode("utf-8") + MESSAGE_TERMINATOR)
            await writer.drain()
            logger.debug(f"Sent response: {response_json}")
        except Exception as e:
            logger.error(f"Error sending response: {e}")

    async def _handle_start_command(
        self, writer: asyncio.StreamWriter, command: StartCommand
    ) -> None:
        """Handle Start command.

        Args:
            writer: StreamWriter to send response through
            command: Start command with output mode
        """
        logger.info(f"Handling Start command (Output: {command.output_mode.value})")

        try:
            await self.pipeline_manager.start_transcription(command.output_mode)
            self.state_manager.set_state(DaemonStateEnum.LISTENING)
            await self._send_response(writer, ResponseWrapper(root=AckResponse()))
        except Exception as e:
            error_msg = f"Failed to start transcription: {e}"
            logger.exception(error_msg)
            self.state_manager.set_error(error_msg)
            await self._send_response(
                writer, ResponseWrapper(root=ErrorResponse(message=error_msg))
            )

    async def _handle_stop_command(self, writer: asyncio.StreamWriter) -> None:
        """Handle Stop command.

        Args:
            writer: StreamWriter to send response through
        """
        logger.info("Handling Stop command")

        try:
            await self.pipeline_manager.stop_transcription()
            if self.state_manager.current_state != DaemonStateEnum.ERROR:
                self.state_manager.set_state(DaemonStateEnum.IDLE)
            await self._send_response(writer, ResponseWrapper(root=AckResponse()))
        except Exception as e:
            error_msg = f"Error stopping transcription: {e}"
            logger.exception(error_msg)
            self.state_manager.set_error(error_msg)
            await self._send_response(
                writer, ResponseWrapper(root=ErrorResponse(message=error_msg))
            )

    async def _handle_status_command(self, writer: asyncio.StreamWriter) -> None:
        """Handle Status command.

        Args:
            writer: StreamWriter to send response through
        """
        logger.debug("Handling Status command")
        state, error = self.state_manager.get_status()
        state_model = DaemonStateModel(state=state, last_error=error)
        response = ResponseWrapper(root=StatusResponse(status=state_model))
        await self._send_response(writer, response)

    async def _handle_shutdown_command(self, writer: asyncio.StreamWriter) -> None:
        """Handle Shutdown command.

        Args:
            writer: StreamWriter to send response through
        """
        logger.info("Handling Shutdown command")

        # Stop any active processing first
        if self.state_manager.current_state != DaemonStateEnum.IDLE:
            await self.pipeline_manager.stop_transcription()

        # Send acknowledgment
        await self._send_response(writer, ResponseWrapper(root=AckResponse()))
        await writer.drain()  # Ensure it's sent

        # Signal shutdown
        self.shutdown_event.set()

    async def _handle_toggle_command(
        self, writer: asyncio.StreamWriter, command: ToggleCommand
    ) -> None:
        """Handle Toggle command.

        Args:
            writer: StreamWriter to send response through
            command: Toggle command with optional output mode
        """
        logger.info(f"Handling Toggle command (Output: {command.output_mode})")

        current_state = self.state_manager.current_state

        if current_state == DaemonStateEnum.IDLE:
            # Start
            # If output mode is None, use default (KEYBOARD)
            mode = command.output_mode or CliOutputMode.KEYBOARD
            try:
                await self.pipeline_manager.start_transcription(mode)
                self.state_manager.set_state(DaemonStateEnum.LISTENING)
                await self._send_response(writer, ResponseWrapper(root=AckResponse()))
            except Exception as e:
                error_msg = f"Failed to toggle start: {e}"
                logger.exception(error_msg)
                self.state_manager.set_error(error_msg)
                await self._send_response(
                    writer, ResponseWrapper(root=ErrorResponse(message=error_msg))
                )

        elif current_state in (DaemonStateEnum.LISTENING, DaemonStateEnum.PROCESSING):
            # Stop
            try:
                await self.pipeline_manager.stop_transcription()
                self.state_manager.set_state(DaemonStateEnum.IDLE)
                await self._send_response(writer, ResponseWrapper(root=AckResponse()))
            except Exception as e:
                error_msg = f"Failed to toggle stop: {e}"
                logger.exception(error_msg)
                self.state_manager.set_error(error_msg)
                await self._send_response(
                    writer, ResponseWrapper(root=ErrorResponse(message=error_msg))
                )
        else:
            # Error state
            error_msg = f"Cannot toggle from state: {current_state}"
            await self._send_response(
                writer, ResponseWrapper(root=ErrorResponse(message=error_msg))
            )

    async def _handle_subscribe_command(self, writer: asyncio.StreamWriter) -> bool:
        """Handle Subscribe command.

        Args:
            writer: StreamWriter to subscribe

        Returns:
            True indicating the client is now subscribed
        """
        logger.info("Handling Subscribe command")
        self._subscribers.add(writer)

        # Send initial status immediately
        state, error = self.state_manager.get_status()
        state_model = DaemonStateModel(state=state, last_error=error)
        notification = ResponseWrapper(root=StateNotification(status=state_model))
        await self._send_response(writer, notification)

        return True

    async def _handle_command(self, writer: asyncio.StreamWriter, message: str) -> bool:
        """Parse and handle a command message.

        Args:
            writer: StreamWriter to send responses through
            message: Command message to parse and handle

        Returns:
            True if connection should be kept alive, False to close it
        """
        try:
            command = CommandWrapper.model_validate_json(message)
            logger.debug(f"Parsed command: {command.model_dump_json()}")

            if isinstance(command.root, StartCommand):
                await self._handle_start_command(writer, command.root)
                return True
            elif isinstance(command.root, StopCommand):
                await self._handle_stop_command(writer)
                return True
            elif isinstance(command.root, StatusCommand):
                await self._handle_status_command(writer)
                return True
            elif isinstance(command.root, ShutdownCommand):
                await self._handle_shutdown_command(writer)
                return False
            elif isinstance(command.root, ToggleCommand):
                await self._handle_toggle_command(writer, command.root)
                return True
            elif isinstance(command.root, SubscribeCommand):
                await self._handle_subscribe_command(writer)
                return True
            else:
                logger.error(f"Unhandled command type: {type(command.root)}")
                await self._send_response(
                    writer,
                    ResponseWrapper(
                        root=ErrorResponse(message="Internal server error")
                    ),
                )
                return True

        except ValidationError as e:
            logger.error(f"Invalid command format: {e}")
            await self._send_response(
                writer,
                ResponseWrapper(
                    root=ErrorResponse(message=f"Invalid command format: {e}")
                ),
            )
            return True

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON: {e}")
            await self._send_response(
                writer,
                ResponseWrapper(
                    root=ErrorResponse(message=f"Invalid JSON format: {e}")
                ),
            )
            return True

        except Exception as e:
            logger.exception("Error handling command")
            await self._send_response(
                writer,
                ResponseWrapper(root=ErrorResponse(message=f"Internal error: {e}")),
            )
            return True

    async def _handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """Handle a client connection.

        Args:
            reader: StreamReader for the client
            writer: StreamWriter for the client
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
                    # Determine timeout based on subscription status
                    timeout = None if writer in self._subscribers else 5.0

                    # Read a line (command should end with newline)
                    data = await asyncio.wait_for(
                        reader.readuntil(MESSAGE_TERMINATOR), timeout=timeout
                    )

                    if not data:  # EOF
                        logger.info(f"Client disconnected (EOF): {peer}")
                        break

                    # Remove terminator and decode
                    message = data.rstrip(MESSAGE_TERMINATOR).decode("utf-8")
                    logger.debug(f"Received from {peer}: {message}")

                    # Handle the command
                    keep_alive = await self._handle_command(writer, message)
                    if not keep_alive:
                        break

                except asyncio.TimeoutError:
                    if writer not in self._subscribers:
                        logger.warning(f"Timeout reading from client {peer}")
                        break
                    # If subscribed, timeout is None, so this shouldn't happen,
                    # but safety check.
                except asyncio.IncompleteReadError:
                    logger.info(f"Client disconnected (incomplete read): {peer}")
                    break
                except ConnectionError as e:
                    logger.warning(f"Connection error with {peer}: {e}")
                    break
                except asyncio.CancelledError:
                    logger.info(f"Client connection cancelled: {peer}")
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

    async def start(self) -> None:
        """Start the IPC server."""
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
        """Stop the IPC server."""
        if not self._server:
            logger.warning("Server not running")
            return

        logger.info("Stopping IPC server...")

        # Explicitly close subscriber connections first to unblock their read loops
        if self._subscribers:
            logger.info(f"Closing {len(self._subscribers)} subscriber connections...")
            for writer in self._subscribers:
                if not writer.is_closing():
                    writer.close()
            self._subscribers.clear()

        # Stop any active processing
        if self.state_manager.current_state != DaemonStateEnum.IDLE:
            await self.pipeline_manager.stop_transcription()

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
