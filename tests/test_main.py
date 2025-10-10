"""Tests for main daemon module."""

import asyncio
import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from handsfreed.main import main
from handsfreed.config import AppConfig


@pytest.fixture
def mock_config():
    """Create a mock config."""
    with patch("handsfreed.main.load_config") as mock_load:
        config = AppConfig.model_construct()  # Create with defaults
        mock_load.return_value = config
        yield config


@pytest.fixture
def mock_handlers():
    """Create mock handlers for the new architecture."""
    # Create a real event for shutdown testing
    shutdown_event = asyncio.Event()

    with (
        patch("handsfreed.main.IPCServer") as mock_ipc,
        patch("handsfreed.main.PWRecordCapture") as mock_audio,
        patch("handsfreed.main.Transcriber") as mock_trans,
        patch("handsfreed.main.OutputHandler") as mock_output,
        patch("handsfreed.main.asyncio.Event", return_value=shutdown_event),
    ):
        # Setup mocked instances
        ipc = AsyncMock()
        audio = AsyncMock()
        trans = AsyncMock()
        output = AsyncMock()

        # Configure mock constructors
        mock_ipc.return_value = ipc
        mock_audio.return_value = audio
        mock_trans.return_value = trans
        mock_output.return_value = output

        # Make transcriber.load_model return True (not async)
        trans.load_model = MagicMock(return_value=True)
        trans.start.return_value = True

        yield {
            "ipc": ipc,
            "audio": audio,
            "trans": trans,
            "output": output,
            "shutdown_event": shutdown_event,
        }


@pytest.mark.asyncio
async def test_main_startup_shutdown(mock_config, mock_handlers):
    """Test normal startup and shutdown flow."""
    main_task = asyncio.create_task(main())

    try:
        # Wait a bit for startup
        await asyncio.sleep(0.1)

        # Trigger shutdown
        mock_handlers["shutdown_event"].set()

        # Wait for main to finish with timeout
        exit_code = await asyncio.wait_for(main_task, timeout=1.0)

        # Check successful exit
        assert exit_code == 0

        # Verify startup sequence
        mock_handlers["trans"].load_model.assert_called_once()
        mock_handlers["trans"].start.assert_awaited_once()
        mock_handlers["output"].start.assert_awaited_once()
        mock_handlers["ipc"].start.assert_awaited_once()

        # Verify shutdown
        mock_handlers["ipc"].stop.assert_awaited_once()
        mock_handlers["trans"].stop.assert_awaited_once()
        mock_handlers["output"].stop.assert_awaited_once()

    finally:
        if not main_task.done():
            main_task.cancel()


@pytest.mark.asyncio
async def test_main_model_load_failure(mock_config, mock_handlers):
    """Test handling of Whisper model load failure."""
    mock_handlers["trans"].load_model.return_value = False

    exit_code = await main()

    assert exit_code == 1
    mock_handlers["ipc"].start.assert_not_awaited()


@pytest.mark.asyncio
async def test_main_transcriber_start_failure(mock_config, mock_handlers):
    """Test handling of transcriber start failure."""
    mock_handlers["trans"].start.return_value = False

    exit_code = await main()

    assert exit_code == 1


@pytest.mark.asyncio
async def test_main_startup_error(mock_config, mock_handlers):
    """Test handling of a fatal error during startup."""
    mock_handlers["ipc"].start.side_effect = RuntimeError("Test error")

    exit_code = await main()

    assert exit_code == 1

    # Verify cleanup is attempted on the components that were started
    mock_handlers["trans"].stop.assert_awaited_once()
    mock_handlers["output"].stop.assert_awaited_once()
    mock_handlers["ipc"].stop.assert_awaited_once()
