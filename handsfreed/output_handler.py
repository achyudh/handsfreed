"""Output execution module for transcribed text."""

import asyncio
import logging
import os
from typing import Literal, Optional, Tuple

from .config import OutputConfig
from .ipc_models import CliOutputMode
from .pipeline import AbstractPipelineConsumerComponent

logger = logging.getLogger(__name__)

# Default commands by session type
DEFAULT_KEYBOARD_WAYLAND = "wtype -"
DEFAULT_KEYBOARD_X11 = "xdotool type --delay 0"
DEFAULT_CLIPBOARD_WAYLAND = "wl-copy"
DEFAULT_CLIPBOARD_X11 = "xclip -selection clipboard"


def get_session_type() -> Literal["wayland", "x11", "unknown"]:
    """Detect the current session type (Wayland/X11/unknown).

    Returns:
        Session type as string: "wayland", "x11", or "unknown"
    """
    session_type = os.environ.get("XDG_SESSION_TYPE", "").lower()

    if session_type == "wayland":
        return "wayland"
    elif session_type == "x11":
        return "x11"
    else:
        return "unknown"


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

    # Get session type
    session = get_session_type()

    # Select command based on mode
    if output_mode == CliOutputMode.KEYBOARD:
        configured_cmd = config.keyboard_command
        mode_str = "keyboard"

        # Get default based on session
        if session == "wayland":
            default_cmd = DEFAULT_KEYBOARD_WAYLAND
        elif session == "x11":
            default_cmd = DEFAULT_KEYBOARD_X11
        else:
            default_cmd = None
    else:  # CLIPBOARD
        configured_cmd = config.clipboard_command
        mode_str = "clipboard"

        # Get default based on session
        if session == "wayland":
            default_cmd = DEFAULT_CLIPBOARD_WAYLAND
        elif session == "x11":
            default_cmd = DEFAULT_CLIPBOARD_X11
        else:
            default_cmd = None

    # Use configured command if available, otherwise use default
    if configured_cmd:
        command_to_run = configured_cmd
    elif default_cmd:
        command_to_run = default_cmd
        logger.debug(f"Using default {mode_str} command for {session}: {default_cmd}")
    else:
        msg = f"No {mode_str} command configured and couldn't determine default for session type: {session}"
        logger.error(msg)
        return False, msg

    logger.debug(f"Executing {mode_str} command: {command_to_run}")
    logger.debug(f"Text length: {len(text)} chars")

    try:
        # Create subprocess with pipes
        process = await asyncio.create_subprocess_shell(
            command_to_run,
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
                    f"Command: {command_to_run}\n"
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
            logger.error(f"Command timed out after {timeout}s: {command_to_run}")
            # Try to kill the process
            try:
                await process.kill()
                await process.wait()
            except ProcessLookupError:
                pass  # Process already finished
            return False, f"Command timed out after {timeout}s"

    except FileNotFoundError:
        msg = f"Command not found: {command_to_run}"
        logger.error(msg)
        return False, msg

    except PermissionError:
        msg = f"Permission denied executing: {command_to_run}"
        logger.error(msg)
        return False, msg

    except Exception as e:
        msg = f"Error executing command: {e}"
        logger.exception(msg)
        return False, msg


class OutputHandler(AbstractPipelineConsumerComponent):
    """Handles output execution for transcribed text."""

    def __init__(
        self,
        config: OutputConfig,
        output_queue: asyncio.Queue,
        stop_event: asyncio.Event,
    ):
        """Initialize output handler.

        Args:
            config: Output configuration containing commands
            output_queue: Queue to receive (text, mode) tuples from.
            stop_event: Event to signal when to stop processing.
        """
        super().__init__(output_queue, None, stop_event)
        self.config = config
        self._needs_leading_space: bool = False

    async def _consume_item(self, item: Tuple[str, CliOutputMode]) -> None:
        """Process a single output request."""
        text, mode = item
        # Apply spacing logic - prepend space if needed
        if text:
            text_to_output = text
            if self._needs_leading_space:
                logger.debug("Prepending space to output")
                text_to_output = " " + text

            # Execute the output command
            success, error = await execute_output_command(
                text_to_output, mode, self.config
            )

            if success:
                # Set flag for next time only if this output succeeded
                self._needs_leading_space = True
            else:
                logger.error(f"Output failed: {error}")
        else:
            logger.warning("Skipping empty text output")

    def reset_spacing_state(self):
        """Reset the spacing flag (e.g., after Start or Stop command).

        This should be called when the transcription is restarted to ensure
        no leading space is added to the first transcription output.
        """
        logger.debug("Resetting output spacing state")
        self._needs_leading_space = False
