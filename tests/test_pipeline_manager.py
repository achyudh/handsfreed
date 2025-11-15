import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest
import pytest_asyncio

from handsfreed.pipeline_manager import PipelineManager
from handsfreed.config import AppConfig, VadConfig
from handsfreed.ipc_models import CliOutputMode
from handsfreed.segmentation.fixed import FixedSegmentationStrategy
from handsfreed.segmentation.vad import VADSegmentationStrategy


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

        manager = PipelineManager(mock_config, stop_event)
        # Replace component instances with mocks
        manager.audio_capture = AsyncMock()
        manager.transcriber = AsyncMock()
        manager.transcriber.load_model = Mock(return_value=True)
        manager.output_handler = AsyncMock()
        manager.output_handler.reset_spacing_state = Mock()
        # The strategy is already mocked by the factory patch
        manager.segmentation_strategy = mock_segmentation_strategy
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


def test_vad_strategy_selection(stop_event):
    """Test that VADSegmentationStrategy is selected when VAD is enabled."""
    config = AppConfig.model_construct()
    config.vad = VadConfig(enabled=True)

    with patch("faster_whisper.vad.get_vad_model", return_value=Mock()):
        manager = PipelineManager(config, stop_event)
        assert isinstance(manager.segmentation_strategy, VADSegmentationStrategy)


def test_vad_strategy_selection_import_error(stop_event):
    """Test that FixedSegmentationStrategy is used when VAD import fails."""
    config = AppConfig.model_construct()
    config.vad = VadConfig(enabled=True)

    with patch("faster_whisper.vad.get_vad_model", side_effect=ImportError):
        manager = PipelineManager(config, stop_event)
        assert isinstance(manager.segmentation_strategy, FixedSegmentationStrategy)


def test_fixed_strategy_selection(stop_event):
    """Test that FixedSegmentationStrategy is selected by default."""
    config = AppConfig.model_construct()
    config.vad = VadConfig(enabled=False)

    manager = PipelineManager(config, stop_event)
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
