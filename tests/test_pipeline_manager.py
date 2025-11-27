import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest
import pytest_asyncio

from handsfreed.pipeline_manager import PipelineManager
from handsfreed.config import AppConfig, VadConfig
from handsfreed.ipc_models import CliOutputMode
from handsfreed.segmentation.fixed import FixedSegmentationStrategy
from handsfreed.segmentation.vad import (
    VADSegmentationStrategy,
    SpeechState,
    SilentState,
)
from handsfreed.state import DaemonStateManager, DaemonStateEnum


@pytest.fixture
def mock_config():
    """Create a mock config."""
    config = AppConfig.model_construct()
    return config


@pytest.fixture
def stop_event():
    """Create a stop event."""
    return asyncio.Event()


@pytest_asyncio.fixture
async def pipeline_manager(mock_config, stop_event):
    """Create a PipelineManager instance with mock components."""
    with (
        patch("handsfreed.pipeline_manager.AudioCapture") as mock_audio,
        patch("handsfreed.pipeline_manager.Transcriber") as mock_trans,
        patch("handsfreed.pipeline_manager.OutputHandler") as mock_output,
        patch(
            "handsfreed.pipeline_manager.create_segmentation_strategy"
        ) as mock_create_strategy,
    ):
        # Mock the return value of the factory
        mock_segmentation_strategy = AsyncMock()
        mock_create_strategy.return_value = mock_segmentation_strategy

        mock_state_manager = Mock(spec=DaemonStateManager)
        manager = PipelineManager(mock_config, stop_event, mock_state_manager)
        # Replace component instances with mocks
        manager.audio_capture = AsyncMock()
        manager.transcriber = AsyncMock()
        manager.transcriber.load_model = Mock(return_value=True)
        manager.output_handler = AsyncMock()
        manager.output_handler.reset_spacing_state = Mock()
        # The strategy is already mocked by the factory patch
        manager.segmentation_strategy = mock_segmentation_strategy
        yield manager


@pytest_asyncio.fixture
async def pipeline_manager_with_real_vad(mock_config, stop_event):
    """Create a PipelineManager instance with a real VADSegmentationStrategy."""
    with (
        patch("handsfreed.pipeline_manager.AudioCapture") as mock_audio,
        patch("handsfreed.pipeline_manager.Transcriber") as mock_trans,
        patch("handsfreed.pipeline_manager.OutputHandler") as mock_output,
        patch(
            "faster_whisper.vad.get_vad_model", return_value=Mock()
        ),  # Mock VAD model
        patch(
            "handsfreed.segmentation.vad.VADSegmentationStrategy._finalize_segment",
            new_callable=AsyncMock,
        ),  # Mock finalize_segment
    ):
        mock_config.vad.enabled = (
            True  # Ensure VAD is enabled for real strategy creation
        )
        mock_state_manager = Mock(spec=DaemonStateManager)

        # Manually create the real segmentation strategy instance
        real_segmentation_strategy = VADSegmentationStrategy(
            raw_audio_queue=asyncio.Queue(),
            transcription_queue=asyncio.Queue(),
            stop_event=stop_event,
            config=mock_config,
            vad_model=Mock(),  # This will be replaced by the patch above
            auto_disable_event=asyncio.Event(),
        )

        with patch(
            "handsfreed.pipeline_manager.create_segmentation_strategy",
            return_value=real_segmentation_strategy,
        ):
            manager = PipelineManager(mock_config, stop_event, mock_state_manager)
            manager.audio_capture = AsyncMock()
            manager.transcriber = AsyncMock()
            manager.transcriber.load_model = Mock(return_value=True)
            manager.output_handler = AsyncMock()
            manager.output_handler.reset_spacing_state = Mock()

            yield manager


@pytest.mark.asyncio
async def test_start_transcription_resets_spacing_state(pipeline_manager):
    """Test start_transcription resets spacing state."""
    await pipeline_manager.start_transcription(CliOutputMode.KEYBOARD)
    pipeline_manager.output_handler.reset_spacing_state.assert_called_once()


@pytest.mark.asyncio
async def test_stop_transcription_resets_spacing_state(pipeline_manager):
    """Test stop_transcription resets spacing state."""
    await pipeline_manager.stop_transcription()
    pipeline_manager.output_handler.reset_spacing_state.assert_called_once()


@pytest.mark.asyncio
async def test_set_active_output_mode_resets_vad_state(pipeline_manager_with_real_vad):
    """Test that setting active output mode to None resets the VAD state."""
    # Simulate being in a speech state
    pipeline_manager_with_real_vad.segmentation_strategy._current_vad_state = (
        SpeechState()
    )

    # Stop transcription
    await pipeline_manager_with_real_vad.segmentation_strategy.set_active_output_mode(
        None
    )

    # Assert that the VAD state is reset to SilentState
    assert isinstance(
        pipeline_manager_with_real_vad.segmentation_strategy._current_vad_state,
        SilentState,
    )
    # And that its _entry_time is None
    assert (
        pipeline_manager_with_real_vad.segmentation_strategy._current_vad_state._entry_time
        is None
    )


def test_vad_strategy_selection(stop_event):
    """Test that VADSegmentationStrategy is selected when VAD is enabled."""
    config = AppConfig.model_construct()
    config.vad = VadConfig(enabled=True)

    with patch("faster_whisper.vad.get_vad_model", return_value=Mock()):
        manager = PipelineManager(config, stop_event, Mock(spec=DaemonStateManager))
        assert isinstance(manager.segmentation_strategy, VADSegmentationStrategy)


def test_vad_strategy_selection_import_error(stop_event):
    """Test that FixedSegmentationStrategy is used when VAD import fails."""
    config = AppConfig.model_construct()
    config.vad = VadConfig(enabled=True)

    with patch("faster_whisper.vad.get_vad_model", side_effect=ImportError):
        manager = PipelineManager(config, stop_event, Mock(spec=DaemonStateManager))
        assert isinstance(manager.segmentation_strategy, FixedSegmentationStrategy)


def test_fixed_strategy_selection(stop_event):
    """Test that FixedSegmentationStrategy is selected by default."""
    config = AppConfig.model_construct()
    config.vad = VadConfig(enabled=False)

    manager = PipelineManager(config, stop_event, Mock(spec=DaemonStateManager))
    assert isinstance(manager.segmentation_strategy, FixedSegmentationStrategy)


@pytest.mark.asyncio
async def test_start_starts_components(pipeline_manager):
    """Test that the start method starts all components."""
    pipeline_manager.transcriber.load_model.return_value = True

    await pipeline_manager.start()

    pipeline_manager.transcriber.load_model.assert_called_once()
    pipeline_manager.transcriber.start.assert_awaited_once()
    pipeline_manager.output_handler.start.assert_awaited_once()
    pipeline_manager.segmentation_strategy.start.assert_awaited_once()
    pipeline_manager.audio_capture.start.assert_awaited_once()


@pytest.mark.asyncio
async def test_stop_stops_components(pipeline_manager):
    """Test that the stop method stops all components."""
    await pipeline_manager.stop()

    pipeline_manager.audio_capture.stop.assert_awaited_once()
    pipeline_manager.segmentation_strategy.stop.assert_awaited_once()
    pipeline_manager.transcriber.stop.assert_awaited_once()
    pipeline_manager.output_handler.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_start_raises_error_on_model_load_failure(pipeline_manager):
    """Test that start raises an error if model loading fails."""
    pipeline_manager.transcriber.load_model.return_value = False

    with pytest.raises(RuntimeError, match="Failed to load Whisper model"):
        await pipeline_manager.start()


@pytest.mark.asyncio
async def test_auto_disable_monitor(pipeline_manager):
    """Test that the auto-disable monitor stops transcription and updates state."""
    # Manually start the task since we aren't calling pipeline_manager.start()
    pipeline_manager._auto_disable_task = asyncio.create_task(
        pipeline_manager._monitor_auto_disable()
    )

    # Setup
    pipeline_manager._auto_disable_event.set()

    # Wait briefly for the background task to react
    # We use wait_for to ensure the test doesn't hang if logic is broken
    try:
        await asyncio.wait_for(pipeline_manager._auto_disable_task, timeout=0.1)
    except asyncio.TimeoutError:
        pass  # Expected, as the loop continues running
    except asyncio.CancelledError:
        pass

    # Verification
    pipeline_manager.output_handler.reset_spacing_state.assert_called()
    pipeline_manager.state_manager.set_state.assert_called_with(DaemonStateEnum.IDLE)
    assert not pipeline_manager._auto_disable_event.is_set()

    # Cleanup task
    pipeline_manager._auto_disable_task.cancel()
    try:
        await pipeline_manager._auto_disable_task
    except asyncio.CancelledError:
        pass
