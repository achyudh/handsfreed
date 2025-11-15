"""Tests for transcriber module."""

import asyncio
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import pytest_asyncio
from faster_whisper import WhisperModel
from handsfreed.config import AppConfig, WhisperConfig
from handsfreed.ipc_models import CliOutputMode
from handsfreed.pipeline import TranscriptionTask
from handsfreed.transcriber import Transcriber


@dataclass
class MockWhisperSegment:
    """Mock segment from faster-whisper."""

    text: str


@dataclass
class MockWhisperInfo:
    """Mock info from faster-whisper."""

    language: str
    language_probability: float
    duration: float


@pytest.fixture
def transcription_queue():
    """Create a transcription queue."""
    return asyncio.Queue()


@pytest.fixture
def output_queue():
    """Create an output queue."""
    return asyncio.Queue()


@pytest.fixture
def config():
    """Create a test config."""
    return AppConfig(
        whisper=WhisperConfig(
            model="tiny.en",
            device="cpu",
            compute_type="float32",
            language="en",
            beam_size=1,
        ),
    )


@pytest.fixture
def mock_model():
    """Create a mock Whisper model."""
    model = MagicMock(spec=WhisperModel)

    def transcribe(*args, **kwargs):
        # Return iterator of segments and info
        segments = [MockWhisperSegment(text="Test transcription")]
        info = MockWhisperInfo(
            language="en",
            language_probability=0.98,
            duration=1.0,
        )
        return iter(segments), info

    model.transcribe = MagicMock(side_effect=transcribe)
    return model


@pytest.fixture
def stop_event():
    """Create a stop event."""
    return asyncio.Event()


@pytest_asyncio.fixture
async def transcriber(config, transcription_queue, output_queue, stop_event):
    """Create a transcriber instance."""
    trans = Transcriber(config, transcription_queue, output_queue, stop_event)
    yield trans
    # Cleanup
    await trans.stop()


def test_load_model_success(transcriber):
    """Test successful model loading."""
    with patch(
        "handsfreed.transcriber.WhisperModel", return_value=MagicMock()
    ) as mock_whisper_model:
        assert transcriber.load_model() is True
        assert transcriber._model is not None

        # Verify cpu_threads parameter is passed
        mock_whisper_model.assert_called_once()
        call_args = mock_whisper_model.call_args[1]
        assert "cpu_threads" in call_args
        assert call_args["cpu_threads"] == 0  # Default value


def test_load_model_failure(transcriber):
    """Test handling of model loading failure."""
    with patch(
        "handsfreed.transcriber.WhisperModel",
        side_effect=RuntimeError("Test error"),
    ):
        assert transcriber.load_model() is False
        assert transcriber._model is None


def test_load_model_already_loaded(transcriber, mock_model):
    """Test loading when model already exists."""
    transcriber._model = mock_model
    assert transcriber.load_model() is True


@pytest.mark.asyncio
async def test_start_without_model(transcriber):
    """Test start fails without model."""
    with pytest.raises(RuntimeError, match="Model not loaded"):
        await transcriber.start()


@pytest.mark.asyncio
async def test_transcription_success_keyboard(
    transcriber, mock_model, transcription_queue, output_queue
):
    transcriber._model = mock_model

    await transcriber.start()

    # Create and send test task
    test_audio = np.zeros(16000, dtype=np.float32)  # 1 second of silence
    task = TranscriptionTask(audio=test_audio, output_mode=CliOutputMode.KEYBOARD)
    await transcription_queue.put(task)

    # Get result
    text, mode = await asyncio.wait_for(output_queue.get(), timeout=1.0)
    assert text == "Test transcription"
    assert mode == CliOutputMode.KEYBOARD


@pytest.mark.asyncio
async def test_transcription_success_clipboard(
    transcriber, mock_model, transcription_queue, output_queue
):
    """Test successful transcription to clipboard."""
    transcriber._model = mock_model

    await transcriber.start()

    # Create and send test task
    test_audio = np.zeros(16000, dtype=np.float32)  # 1 second of silence
    task = TranscriptionTask(audio=test_audio, output_mode=CliOutputMode.CLIPBOARD)
    await transcription_queue.put(task)

    # Get result
    text, mode = await asyncio.wait_for(output_queue.get(), timeout=1.0)
    assert text == "Test transcription"
    assert mode == CliOutputMode.CLIPBOARD


@pytest.mark.asyncio
async def test_transcription_error(
    transcriber, mock_model, transcription_queue, output_queue
):
    """Test handling of transcription error."""
    transcriber._model = mock_model
    mock_model.transcribe.side_effect = RuntimeError("Test error")

    await transcriber.start()

    # Create and send test task
    test_audio = np.zeros(16000, dtype=np.float32)
    task = TranscriptionTask(audio=test_audio, output_mode=CliOutputMode.KEYBOARD)
    await transcription_queue.put(task)

    # No result should be put on output queue
    await asyncio.sleep(0.1)  # Give time for processing
    assert output_queue.empty()


@pytest.mark.asyncio
async def test_transcription_empty_result(
    transcriber, mock_model, transcription_queue, output_queue
):
    """Test handling of empty transcription result."""
    transcriber._model = mock_model

    # Return empty segments
    mock_model.transcribe = MagicMock(
        return_value=(iter([]), MockWhisperInfo("en", 0.98, 1.0))
    )

    await transcriber.start()

    # Create and send test task
    test_audio = np.zeros(16000, dtype=np.float32)
    task = TranscriptionTask(audio=test_audio, output_mode=CliOutputMode.KEYBOARD)
    await transcription_queue.put(task)

    # No result should be put on output queue
    await asyncio.sleep(0.1)  # Give time for processing
    assert output_queue.empty()


@pytest.mark.asyncio
async def test_multiple_start_stop(transcriber, mock_model):
    """Test multiple start/stop cycles."""
    transcriber._model = mock_model

    # First cycle
    await transcriber.start()
    task1 = transcriber._task
    await transcriber.stop()

    # Second cycle
    await transcriber.start()
    task2 = transcriber._task
    await transcriber.stop()

    assert task1 is not task2  # Should be different task objects
    assert transcriber._task is None  # Should be cleaned up
