"""Tests for main daemon module."""

import asyncio
import pytest
from unittest.mock import patch, AsyncMock, MagicMock, Mock

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
    """Create mock handlers."""
    # Create a real event for shutdown testing
    shutdown_event = asyncio.Event()
    stop_event = asyncio.Event()

    with (
        patch("handsfreed.main.IPCServer") as mock_ipc,
        patch("handsfreed.main.PipelineManager") as mock_pm,
        patch("handsfreed.main.asyncio.Event") as mock_event,
    ):
        # Configure event mock to return our events
        mock_event.side_effect = [stop_event, shutdown_event]

        # Setup mocked instances
        ipc = AsyncMock()
        pm = AsyncMock()

        # Configure mock constructors
        mock_ipc.return_value = ipc
        mock_pm.return_value = pm

        yield {
            "ipc": ipc,
            "pm": pm,
            "shutdown_event": shutdown_event,
            "stop_event": stop_event,
        }


@pytest.mark.asyncio
async def test_main_startup_shutdown(mock_config, mock_handlers):
    """Test normal startup and shutdown flow."""
    # Start main in a task so we can trigger shutdown
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
        mock_handlers["pm"].start.assert_awaited_once()
        mock_handlers["ipc"].start.assert_awaited_once()

        # Verify shutdown
        mock_handlers["ipc"].stop.assert_awaited_once()
        assert mock_handlers["stop_event"].is_set()
        mock_handlers["pm"].stop.assert_awaited_once()

    except asyncio.TimeoutError:
        main_task.cancel()
        await asyncio.sleep(0.1)  # Give cancel time to process
        raise
    except:  # noqa: E722
        if not main_task.done():
            main_task.cancel()
            await asyncio.sleep(0.1)
        raise


@pytest.mark.asyncio
async def test_main_pipeline_manager_start_failure(mock_config, mock_handlers):
    """Test handling of PipelineManager start failure."""
    # Make pipeline manager start fail
    mock_handlers["pm"].start.side_effect = RuntimeError("Test error")

    # Run main (should return immediately on failure)
    exit_code = await main()

    # Check error exit
    assert exit_code == 1

    # Verify nothing else was started
    mock_handlers["ipc"].start.assert_not_awaited()


@pytest.mark.asyncio
async def test_main_startup_error(mock_config, mock_handlers):
    """Test handling of startup error."""
    # Make IPC server start raise error
    mock_handlers["ipc"].start.side_effect = RuntimeError("Test error")

    # Run main (should return on error)
    exit_code = await main()

    # Check error exit
    assert exit_code == 1

    # Verify cleanup attempted
    mock_handlers["stop_event"].is_set()
    mock_handlers["pm"].stop.assert_awaited_once()
    mock_handlers["ipc"].stop.assert_awaited_once()
