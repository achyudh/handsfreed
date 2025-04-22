"""Output execution module for transcribed text."""

import asyncio
import logging
from typing import Tuple, Optional

from .config import OutputConfig
from .ipc_models import CliOutputMode

logger = logging.getLogger(__name__)


async def execute_output_command(
    text: str, output_mode: CliOutputMode, config: OutputConfig, timeout: float = 5.0
) -> Tuple[bool, Optional[str]]:
    """Execute the configured output command.

    Args:
        text: The text to output
        output_mode: Which output mode to use (keyboard/clipboard)
        config: Output configuration containing the commands
        timeout: Maximum time to wait for command execution

    Returns:
        Tuple of (success, error_message)
        - success: True if command executed successfully
        - error_message: Error details if command failed, None otherwise
    """
    if not text:
        logger.warning("Skipping output for empty text")
        return True, None

    # Select command based on mode
    if output_mode == CliOutputMode.KEYBOARD:
        command = config.keyboard_command
        mode_str = "keyboard"
    else:  # CLIPBOARD
        command = config.clipboard_command
        mode_str = "clipboard"

    # Validate command
    if not command:
        msg = f"No command configured for {mode_str} output"
        logger.error(msg)
        return False, msg

    logger.debug(f"Executing {mode_str} command: {command}")
    logger.debug(f"Text length: {len(text)} chars")

    try:
        # Create subprocess with pipes
        process = await asyncio.create_subprocess_shell(
            command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            # Send text and wait for completion with timeout
            stdout, stderr = await asyncio.wait_for(
                process.communicate(text.encode("utf-8")), timeout=timeout
            )

            if process.returncode != 0:
                # Log command output on error
                error_msg = (
                    f"Command failed with code {process.returncode}:\n"
                    f"Command: {command}\n"
                    f"Stderr: {stderr.decode('utf-8', errors='replace')}"
                )
                if stdout:
                    error_msg += f"\nStdout: {stdout.decode('utf-8', errors='replace')}"
                logger.error(error_msg)
                return False, error_msg

            # Log stdout if any (some commands might output status/info)
            if stdout:
                logger.debug(
                    f"Command stdout: {stdout.decode('utf-8', errors='replace')}"
                )

            return True, None

        except asyncio.TimeoutError:
            logger.error(f"Command timed out after {timeout}s: {command}")
            # Try to kill the process
            try:
                await process.kill()
                await process.wait()
            except ProcessLookupError:
                pass  # Process already finished
            return False, f"Command timed out after {timeout}s"

    except FileNotFoundError:
        msg = f"Command not found: {command}"
        logger.error(msg)
        return False, msg

    except PermissionError:
        msg = f"Permission denied executing: {command}"
        logger.error(msg)
        return False, msg

    except Exception as e:
        msg = f"Error executing command: {e}"
        logger.exception(msg)
        return False, msg


class OutputHandler:
    """Handles output execution for transcribed text."""

    def __init__(self, config: OutputConfig):
        """Initialize output handler.

        Args:
            config: Output configuration containing commands
        """
        self.config = config
        self._task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()

    async def _output_loop(self, input_queue: asyncio.Queue) -> None:
        """Process output requests from queue.

        Args:
            input_queue: Queue containing (text, mode) tuples
        """
        logger.info("Output handler started")

        while not self._stop_event.is_set():
            try:
                # Get next output request (with timeout to check stop event)
                try:
                    text, mode = await asyncio.wait_for(input_queue.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue

                try:
                    # Execute the output command
                    success, error = await execute_output_command(
                        text, mode, self.config
                    )
                    if not success:
                        logger.error(f"Output failed: {error}")

                finally:
                    # Always mark task as done
                    input_queue.task_done()

            except asyncio.CancelledError:
                logger.info("Output handler cancelled")
                break
            except Exception as e:
                logger.exception(f"Error in output handler: {e}")
                # Avoid tight loop on unexpected errors
                await asyncio.sleep(0.5)

        logger.info("Output handler stopped")

    async def start(self, input_queue: asyncio.Queue) -> None:
        """Start the output handler.

        Args:
            input_queue: Queue to receive (text, mode) tuples from
        """
        if self._task and not self._task.done():
            logger.warning("Output handler already running")
            return

        logger.info("Starting output handler")
        self._stop_event.clear()
        self._task = asyncio.create_task(self._output_loop(input_queue))

    async def stop(self) -> None:
        """Stop the output handler."""
        if not self._task or self._task.done():
            logger.warning("Output handler not running")
            return

        logger.info("Stopping output handler")
        self._stop_event.set()

        try:
            # Wait for task to finish with timeout
            await asyncio.wait_for(self._task, timeout=5.0)
            logger.info("Output handler stopped gracefully")
        except asyncio.TimeoutError:
            logger.warning("Timeout waiting for output handler to stop")
            self._task.cancel()
            await asyncio.sleep(0.1)  # Give cancel time to process
        except Exception as e:
            logger.exception(f"Error stopping output handler: {e}")
        finally:
            self._task = None
