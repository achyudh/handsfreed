"""Tests for output execution module."""

import asyncio
import os
import pytest
import pytest_asyncio
from unittest.mock import patch, AsyncMock

from handsfreed.config import OutputConfig
from handsfreed.ipc_models import CliOutputMode
from handsfreed.output_handler import (
    execute_output_command,
    OutputHandler,
    get_session_type,
    DEFAULT_KEYBOARD_WAYLAND,
    DEFAULT_KEYBOARD_X11,
    DEFAULT_CLIPBOARD_WAYLAND,
    DEFAULT_CLIPBOARD_X11,
)


@pytest.fixture
def config():
    """Create test output config."""
    return OutputConfig(
        keyboard_command="test-keyboard-cmd",
        clipboard_command="test-clipboard-cmd",
    )


@pytest.fixture
def empty_config():
    """Create output config with no commands set."""
    return OutputConfig()


@pytest.fixture
def output_queue():
    """Create output queue."""
    return asyncio.Queue()


@pytest.fixture
def stop_event():
    """Create a stop event."""
    return asyncio.Event()


@pytest_asyncio.fixture
async def handler(config, output_queue, stop_event):
    """Create output handler."""
    handler = OutputHandler(config, output_queue, stop_event)
    yield handler
    await handler.stop()


def test_get_session_type_wayland():
    """Test Wayland detection."""
    with patch.dict(os.environ, {"XDG_SESSION_TYPE": "wayland"}):
        assert get_session_type() == "wayland"


def test_get_session_type_x11():
    """Test X11 detection."""
    with patch.dict(os.environ, {"XDG_SESSION_TYPE": "x11"}):
        assert get_session_type() == "x11"


def test_get_session_type_unknown():
    """Test unknown session type."""
    with patch.dict(os.environ, {"XDG_SESSION_TYPE": ""}):
        assert get_session_type() == "unknown"

    # Test with session type not set at all
    with patch.dict(os.environ, clear=True):
        assert get_session_type() == "unknown"


@pytest.mark.asyncio
async def test_execute_output_keyboard(config):
    """Test keyboard output execution with configured command."""
    mock_process = AsyncMock()
    mock_process.communicate = AsyncMock(return_value=(b"", b""))
    mock_process.returncode = 0

    with patch(
        "asyncio.create_subprocess_shell", return_value=mock_process
    ) as mock_create:
        success, error = await execute_output_command(
            "test text", CliOutputMode.KEYBOARD, config
        )

        assert success is True
        assert error is None
        mock_create.assert_called_once_with(
            "test-keyboard-cmd",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        mock_process.communicate.assert_called_once_with(b"test text")


@pytest.mark.asyncio
async def test_execute_output_clipboard(config):
    """Test clipboard output execution with configured command."""
    mock_process = AsyncMock()
    mock_process.communicate = AsyncMock(return_value=(b"", b""))
    mock_process.returncode = 0

    with patch("asyncio.create_subprocess_shell", return_value=mock_process):
        success, error = await execute_output_command(
            "test text", CliOutputMode.CLIPBOARD, config
        )

        assert success is True
        assert error is None
        mock_process.communicate.assert_called_once_with(b"test text")


@pytest.mark.asyncio
async def test_execute_output_empty_text(config):
    """Test handling of empty text."""
    with patch("asyncio.create_subprocess_shell") as mock_create:
        success, error = await execute_output_command(
            "", CliOutputMode.KEYBOARD, config
        )

        assert success is True  # Empty text is not an error
        assert error is None
        mock_create.assert_not_called()


@pytest.mark.asyncio
async def test_execute_output_command_not_found(config):
    """Test handling of non-existent command."""
    with patch(
        "asyncio.create_subprocess_shell", side_effect=FileNotFoundError()
    ) as mock_create:
        success, error = await execute_output_command(
            "test text", CliOutputMode.KEYBOARD, config
        )

        assert success is False
        assert "Command not found" in error
        mock_create.assert_called_once()


@pytest.mark.asyncio
async def test_execute_output_timeout(config):
    """Test command timeout handling."""
    mock_process = AsyncMock()
    mock_process.communicate = AsyncMock(side_effect=asyncio.TimeoutError())
    mock_process.kill = AsyncMock()
    mock_process.wait = AsyncMock()

    with patch("asyncio.create_subprocess_shell", return_value=mock_process):
        success, error = await execute_output_command(
            "test text",
            CliOutputMode.KEYBOARD,
            config,
            timeout=0.1,  # Short timeout for test
        )

        assert success is False
        assert "timed out" in error
        mock_process.kill.assert_called_once()


@pytest.mark.asyncio
async def test_execute_output_failure(config):
    """Test command failure handling."""
    mock_process = AsyncMock()
    mock_process.communicate = AsyncMock(return_value=(b"", b"test error"))
    mock_process.returncode = 1

    with patch("asyncio.create_subprocess_shell", return_value=mock_process):
        success, error = await execute_output_command(
            "test text", CliOutputMode.KEYBOARD, config
        )

        assert success is False
        assert "Command failed with code 1" in error
        assert "test error" in error


@pytest.mark.asyncio
async def test_execute_output_wayland_default_keyboard(empty_config):
    """Test default wayland keyboard command is used when config is empty."""
    mock_process = AsyncMock()
    mock_process.communicate = AsyncMock(return_value=(b"", b""))
    mock_process.returncode = 0

    with patch.dict(os.environ, {"XDG_SESSION_TYPE": "wayland"}):
        with patch(
            "asyncio.create_subprocess_shell", return_value=mock_process
        ) as mock_create:
            success, error = await execute_output_command(
                "test text", CliOutputMode.KEYBOARD, empty_config
            )

            assert success is True
            assert error is None
            mock_create.assert_called_once_with(
                DEFAULT_KEYBOARD_WAYLAND,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )


@pytest.mark.asyncio
async def test_execute_output_wayland_default_clipboard(empty_config):
    """Test default wayland clipboard command is used when config is empty."""
    mock_process = AsyncMock()
    mock_process.communicate = AsyncMock(return_value=(b"", b""))
    mock_process.returncode = 0

    with patch.dict(os.environ, {"XDG_SESSION_TYPE": "wayland"}):
        with patch(
            "asyncio.create_subprocess_shell", return_value=mock_process
        ) as mock_create:
            success, error = await execute_output_command(
                "test text", CliOutputMode.CLIPBOARD, empty_config
            )

            assert success is True
            assert error is None
            mock_create.assert_called_once_with(
                DEFAULT_CLIPBOARD_WAYLAND,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )


@pytest.mark.asyncio
async def test_execute_output_x11_default_keyboard(empty_config):
    """Test default x11 keyboard command is used when config is empty."""
    mock_process = AsyncMock()
    mock_process.communicate = AsyncMock(return_value=(b"", b""))
    mock_process.returncode = 0

    with patch.dict(os.environ, {"XDG_SESSION_TYPE": "x11"}):
        with patch(
            "asyncio.create_subprocess_shell", return_value=mock_process
        ) as mock_create:
            success, error = await execute_output_command(
                "test text", CliOutputMode.KEYBOARD, empty_config
            )

            assert success is True
            assert error is None
            mock_create.assert_called_once_with(
                DEFAULT_KEYBOARD_X11,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )


@pytest.mark.asyncio
async def test_execute_output_x11_default_clipboard(empty_config):
    """Test default x11 clipboard command is used when config is empty."""
    mock_process = AsyncMock()
    mock_process.communicate = AsyncMock(return_value=(b"", b""))
    mock_process.returncode = 0

    with patch.dict(os.environ, {"XDG_SESSION_TYPE": "x11"}):
        with patch(
            "asyncio.create_subprocess_shell", return_value=mock_process
        ) as mock_create:
            success, error = await execute_output_command(
                "test text", CliOutputMode.CLIPBOARD, empty_config
            )

            assert success is True
            assert error is None
            mock_create.assert_called_once_with(
                DEFAULT_CLIPBOARD_X11,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )


@pytest.mark.asyncio
async def test_execute_output_unknown_session_type(empty_config):
    """Test error when session type is unknown and config is empty."""
    with patch.dict(os.environ, {"XDG_SESSION_TYPE": "unknown"}):
        success, error = await execute_output_command(
            "test text", CliOutputMode.KEYBOARD, empty_config
        )

        assert success is False
        assert "No keyboard command configured and couldn't determine default" in error


@pytest.mark.asyncio
async def test_configured_command_overrides_default(empty_config):
    """Test that configured command overrides default regardless of session."""
    mock_process = AsyncMock()
    mock_process.communicate = AsyncMock(return_value=(b"", b""))
    mock_process.returncode = 0

    # Set a command in the empty config
    empty_config.keyboard_command = "custom-keyboard-cmd"

    # Even with Wayland session, should use configured command
    with patch.dict(os.environ, {"XDG_SESSION_TYPE": "wayland"}):
        with patch(
            "asyncio.create_subprocess_shell", return_value=mock_process
        ) as mock_create:
            success, error = await execute_output_command(
                "test text", CliOutputMode.KEYBOARD, empty_config
            )

            assert success is True
            assert error is None
            mock_create.assert_called_once_with(
                "custom-keyboard-cmd",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )


@pytest.mark.asyncio
async def test_handler_start_stop(handler, output_queue):
    """Test output handler start/stop."""
    await handler.start()
    assert handler._task is not None
    assert not handler._task.done()

    await handler.stop()
    assert handler._task is None


@pytest.mark.asyncio
async def test_handler_process_output(handler, output_queue):
    """Test output handler processes queue items."""
    mock_execute = AsyncMock(return_value=(True, None))

    with patch("handsfreed.output_handler.execute_output_command", mock_execute):
        await handler.start()

        # Send test output request
        await output_queue.put(("test text", CliOutputMode.KEYBOARD))
        await asyncio.sleep(0.1)  # Give time to process

        mock_execute.assert_called_once_with(
            "test text", CliOutputMode.KEYBOARD, handler.config
        )
        assert output_queue.empty()


@pytest.mark.asyncio
async def test_handler_multiple_outputs(handler, output_queue):
    """Test handler processes multiple outputs."""
    mock_execute = AsyncMock(return_value=(True, None))

    with patch("handsfreed.output_handler.execute_output_command", mock_execute):
        await handler.start()

        # Send multiple output requests
        await output_queue.put(("text1", CliOutputMode.KEYBOARD))
        await output_queue.put(("text2", CliOutputMode.CLIPBOARD))
        await asyncio.sleep(0.2)  # Give time to process both

        assert mock_execute.call_count == 2
        assert output_queue.empty()


@pytest.mark.asyncio
async def test_spacing_state_reset(handler, output_queue):
    """Test reset_spacing_state functionality."""
    mock_execute = AsyncMock()
    mock_execute.side_effect = [(True, None), (True, None), (True, None)]

    with patch("handsfreed.output_handler.execute_output_command", mock_execute):
        await handler.start()

        # First text should NOT have space (default state)
        await output_queue.put(("Text1", CliOutputMode.KEYBOARD))
        await asyncio.sleep(0.1)
        mock_execute.assert_called_with("Text1", CliOutputMode.KEYBOARD, handler.config)

        # Second text SHOULD have space
        await output_queue.put(("Text2", CliOutputMode.KEYBOARD))
        await asyncio.sleep(0.1)
        mock_execute.assert_called_with(
            " Text2", CliOutputMode.KEYBOARD, handler.config
        )

        # Reset state
        handler.reset_spacing_state()

        # Third text should NOT have space (after reset)
        await output_queue.put(("Text3", CliOutputMode.KEYBOARD))
        await asyncio.sleep(0.1)
        mock_execute.assert_called_with("Text3", CliOutputMode.KEYBOARD, handler.config)


@pytest.mark.asyncio
async def test_spacing_state_on_failed_output(handler, output_queue):
    """Test that spacing state is not updated when output fails."""
    mock_execute = AsyncMock()
    mock_execute.side_effect = [(True, None), (False, "Error"), (True, None)]

    with patch("handsfreed.output_handler.execute_output_command", mock_execute):
        await handler.start()

        # First text succeeds - sets needs_leading_space to True
        await output_queue.put(("Text1", CliOutputMode.KEYBOARD))
        await asyncio.sleep(0.1)
        mock_execute.assert_called_with("Text1", CliOutputMode.KEYBOARD, handler.config)

        # Second text fails - needs_leading_space should remain True
        await output_queue.put(("Text2", CliOutputMode.KEYBOARD))
        await asyncio.sleep(0.1)
        mock_execute.assert_called_with(
            " Text2", CliOutputMode.KEYBOARD, handler.config
        )

        # Third text should still have space (since second output failed)
        await output_queue.put(("Text3", CliOutputMode.KEYBOARD))
        await asyncio.sleep(0.1)
        mock_execute.assert_called_with(
            " Text3", CliOutputMode.KEYBOARD, handler.config
        )
