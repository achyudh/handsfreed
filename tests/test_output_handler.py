"""Tests for output execution module."""

import asyncio
import pytest
import pytest_asyncio
from unittest.mock import patch, AsyncMock, MagicMock

from handsfreed.config import OutputConfig
from handsfreed.ipc_models import CliOutputMode
from handsfreed.output_handler import execute_output_command, OutputHandler


@pytest.fixture
def config():
    """Create test output config."""
    return OutputConfig(
        keyboard_command="xdotool type --delay 0",
        clipboard_command="wl-copy",
    )


@pytest.fixture
def output_queue():
    """Create output queue."""
    return asyncio.Queue()


@pytest_asyncio.fixture
async def handler(config):
    """Create output handler."""
    handler = OutputHandler(config)
    yield handler
    if handler._task:
        await handler.stop()


@pytest.mark.asyncio
async def test_execute_output_keyboard(config):
    """Test keyboard output execution."""
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
            "xdotool type --delay 0",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        mock_process.communicate.assert_called_once_with(b"test text")


@pytest.mark.asyncio
async def test_execute_output_clipboard(config):
    """Test clipboard output execution."""
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
async def test_handler_start_stop(handler, output_queue):
    """Test output handler start/stop."""
    await handler.start(output_queue)
    assert handler._task is not None
    assert not handler._task.done()

    await handler.stop()
    assert handler._task is None
    assert handler._stop_event.is_set()


@pytest.mark.asyncio
async def test_handler_process_output(handler, output_queue):
    """Test output handler processes queue items."""
    mock_execute = AsyncMock(return_value=(True, None))

    with patch("handsfreed.output_handler.execute_output_command", mock_execute):
        await handler.start(output_queue)

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
        await handler.start(output_queue)

        # Send multiple output requests
        await output_queue.put(("text1", CliOutputMode.KEYBOARD))
        await output_queue.put(("text2", CliOutputMode.CLIPBOARD))
        await asyncio.sleep(0.2)  # Give time to process both

        assert mock_execute.call_count == 2
        assert output_queue.empty()


@pytest.mark.asyncio
async def test_handler_output_failure(handler, output_queue):
    """Test handler continues after output failure."""
    mock_execute = AsyncMock(side_effect=[(False, "error"), (True, None)])

    with patch("handsfreed.output_handler.execute_output_command", mock_execute):
        await handler.start(output_queue)

        # Send two outputs, first fails
        await output_queue.put(("fail", CliOutputMode.KEYBOARD))
        await output_queue.put(("success", CliOutputMode.KEYBOARD))
        await asyncio.sleep(0.2)  # Give time to process both

        assert mock_execute.call_count == 2  # Should process both
        assert output_queue.empty()  # Queue should be cleared
