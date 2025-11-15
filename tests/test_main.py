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
        patch("handsfreed.main.AudioCapture") as mock_audio,
        patch("handsfreed.main.Transcriber") as mock_trans,
        patch("handsfreed.main.OutputHandler") as mock_output,
        patch("handsfreed.main.FixedSegmentationStrategy") as mock_fixed_strategy,
        patch("handsfreed.main.VADSegmentationStrategy") as mock_vad_strategy,
        patch("handsfreed.main.asyncio.Event") as mock_event,
    ):
        # Configure event mock to return our events
        mock_event.side_effect = [stop_event, shutdown_event]

        # Setup mocked instances
        ipc = AsyncMock()
        audio = AsyncMock()
        trans = AsyncMock()
        output = AsyncMock()
        fixed_strategy_instance = AsyncMock()
        fixed_strategy_instance.process = AsyncMock(return_value=None)
        vad_strategy_instance = AsyncMock()
        vad_strategy_instance.process = AsyncMock(return_value=None)


        # Configure mock constructors
        mock_ipc.return_value = ipc
        mock_audio.return_value = audio
        mock_trans.return_value = trans
        mock_output.return_value = output
        mock_fixed_strategy.return_value = fixed_strategy_instance
        mock_vad_strategy.return_value = vad_strategy_instance

        # Make transcriber.load_model return True (not async)
        trans.load_model = MagicMock(return_value=True)

        yield {
            "ipc": ipc,
            "audio": audio,
            "trans": trans,
            "output": output,
            "fixed_strategy": fixed_strategy_instance,
            "vad_strategy": vad_strategy_instance,
            "shutdown_event": shutdown_event,
            "stop_event": stop_event,
            "mock_fixed_strategy_class": mock_fixed_strategy,
            "mock_vad_strategy_class": mock_vad_strategy,
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
        mock_handlers["trans"].load_model.assert_called_once()
        mock_handlers["trans"].start.assert_awaited_once()
        mock_handlers["output"].start.assert_awaited_once()
        mock_handlers["fixed_strategy"].start.assert_awaited_once()
        mock_handlers["ipc"].start.assert_awaited_once()

        # Verify shutdown
        mock_handlers["ipc"].stop.assert_awaited_once()
        assert mock_handlers["stop_event"].is_set()
        mock_handlers["fixed_strategy"].stop.assert_awaited_once()
        mock_handlers["trans"].stop.assert_awaited_once()
        mock_handlers["output"].stop.assert_awaited_once()

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
async def test_main_model_load_failure(mock_config, mock_handlers):
    """Test handling of Whisper model load failure."""
    # Make model loading fail
    mock_handlers["trans"].load_model.return_value = False

    # Run main (should return immediately on model load failure)
    exit_code = await main()

    # Check error exit
    assert exit_code == 1

    # Verify nothing was started
    mock_handlers["ipc"].start.assert_not_awaited()
    mock_handlers["fixed_strategy"].start.assert_not_awaited()


@pytest.mark.asyncio
async def test_main_transcriber_start_failure(mock_config, mock_handlers):
    """Test handling of transcriber start failure."""
    # Make transcriber start fail
    mock_handlers["trans"].start.side_effect = RuntimeError("Test error")

    # Run main (should return on transcriber start failure)
    exit_code = await main()

    # Check error exit
    assert exit_code == 1


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
    mock_handlers["fixed_strategy"].stop.assert_awaited_once()
    mock_handlers["trans"].stop.assert_awaited_once()
    mock_handlers["output"].stop.assert_awaited_once()
    mock_handlers["ipc"].stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_vad_strategy_selection(mock_config, mock_handlers):
    """Test that VADSegmentationStrategy is selected when VAD is enabled."""
    mock_config.vad.enabled = True
    mock_config.vad.min_silence_duration_ms = 1000 # Ensure VAD config is valid

    with patch("handsfreed.main.get_vad_model") as mock_get_vad_model:
        mock_get_vad_model.return_value = Mock() # Mock a successful VAD model load

        # Start main briefly
        main_task = asyncio.create_task(main())
        await asyncio.sleep(0.1)
        mock_handlers["shutdown_event"].set()
        await main_task

        # Verify VADSegmentationStrategy was created
        mock_handlers["mock_vad_strategy_class"].assert_called_once()
        mock_handlers["vad_strategy"].start.assert_awaited_once()
        mock_handlers["mock_fixed_strategy_class"].assert_not_called()
        mock_get_vad_model.assert_called_once()


@pytest.mark.asyncio
async def test_strategy_selection(mock_config, mock_handlers):
    """Test that FixedSegmentationStrategy is selected by default."""
    # Ensure VAD is disabled in config for this test
    mock_config.vad.enabled = False

    # Start main briefly
    main_task = asyncio.create_task(main())
    await asyncio.sleep(0.1)
    mock_handlers["shutdown_event"].set()
    await main_task

    # Verify FixedSegmentationStrategy was created
    mock_handlers["mock_fixed_strategy_class"].assert_called_once()
    mock_handlers["fixed_strategy"].start.assert_awaited_once()
    mock_handlers["mock_vad_strategy_class"].assert_not_called()
