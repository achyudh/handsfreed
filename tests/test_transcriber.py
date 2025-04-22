"""Tests for transcriber module."""

import asyncio
import pytest
import pytest_asyncio
import numpy as np
from unittest.mock import MagicMock, patch, AsyncMock
from dataclasses import dataclass
from typing import List, Iterator

from faster_whisper import WhisperModel

from handsfreed.config import AppConfig, WhisperConfig, VadConfig
from handsfreed.ipc_models import CliOutputMode
from handsfreed.transcriber import Transcriber, TranscriptionResult


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
def audio_queue():
    """Create an audio queue."""
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
            vad_filter=True,
        ),
        vad=VadConfig(
            threshold=0.5,
            min_speech_duration_ms=250,
            min_silence_duration_ms=1500,
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


@pytest_asyncio.fixture
async def transcriber(config, audio_queue, output_queue):
    """Create a transcriber instance."""
    trans = Transcriber(config, audio_queue, output_queue)
    yield trans
    # Cleanup
    if trans._transcription_task:
        await trans.stop()


def test_load_model_success(transcriber):
    """Test successful model loading."""
    with patch("handsfreed.transcriber.WhisperModel", return_value=MagicMock()):
        assert transcriber.load_model() is True
        assert transcriber._model is not None


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


def test_set_output_mode(transcriber):
    """Test setting output mode."""
    assert transcriber._current_output_mode is None

    transcriber.set_current_output_mode(CliOutputMode.KEYBOARD)
    assert transcriber._current_output_mode == CliOutputMode.KEYBOARD

    transcriber.set_current_output_mode(None)
    assert transcriber._current_output_mode is None


@pytest.mark.asyncio
async def test_start_without_model(transcriber):
    """Test start fails without model."""
    assert await transcriber.start() is False


@pytest.mark.asyncio
async def test_transcription_without_output_mode(
    transcriber, mock_model, audio_queue, output_queue
):
    """Test transcription skips when no output mode set."""
    transcriber._model = mock_model

    assert await transcriber.start() is True

    # Send test audio (without setting output mode)
    await audio_queue.put(np.zeros(16000, dtype=np.float32))
    await asyncio.sleep(0.1)  # Give time for processing

    assert mock_model.transcribe.called  # Should still attempt transcription
    assert output_queue.empty()  # But should not output result


@pytest.mark.asyncio
async def test_transcription_success_keyboard(
    transcriber, mock_model, audio_queue, output_queue
):
    """Test successful transcription to keyboard."""
    transcriber._model = mock_model
    transcriber.set_current_output_mode(CliOutputMode.KEYBOARD)

    assert await transcriber.start() is True

    # Send test audio
    test_audio = np.zeros(16000, dtype=np.float32)  # 1 second of silence
    await audio_queue.put(test_audio)

    # Get result
    text, mode = await asyncio.wait_for(output_queue.get(), timeout=1.0)
    assert text == "Test transcription"
    assert mode == CliOutputMode.KEYBOARD


@pytest.mark.asyncio
async def test_transcription_success_clipboard(
    transcriber, mock_model, audio_queue, output_queue
):
    """Test successful transcription to clipboard."""
    transcriber._model = mock_model
    transcriber.set_current_output_mode(CliOutputMode.CLIPBOARD)

    assert await transcriber.start() is True

    # Send test audio
    test_audio = np.zeros(16000, dtype=np.float32)
    await audio_queue.put(test_audio)

    # Get result
    text, mode = await asyncio.wait_for(output_queue.get(), timeout=1.0)
    assert text == "Test transcription"
    assert mode == CliOutputMode.CLIPBOARD


@pytest.mark.asyncio
async def test_transcription_error(transcriber, mock_model, audio_queue, output_queue):
    """Test handling of transcription error."""
    transcriber._model = mock_model
    mock_model.transcribe.side_effect = RuntimeError("Test error")
    transcriber.set_current_output_mode(CliOutputMode.KEYBOARD)

    await transcriber.start()
    await audio_queue.put(np.zeros(16000, dtype=np.float32))

    # No result should be put on output queue
    await asyncio.sleep(0.1)  # Give time for processing
    assert output_queue.empty()


@pytest.mark.asyncio
async def test_transcription_empty_result(
    transcriber, mock_model, audio_queue, output_queue
):
    """Test handling of empty transcription result."""
    transcriber._model = mock_model
    transcriber.set_current_output_mode(CliOutputMode.KEYBOARD)

    # Return empty segments
    mock_model.transcribe = MagicMock(
        return_value=(iter([]), MockWhisperInfo("en", 0.98, 1.0))
    )

    await transcriber.start()
    await audio_queue.put(np.zeros(16000, dtype=np.float32))

    # No result should be put on output queue
    await asyncio.sleep(0.1)  # Give time for processing
    assert output_queue.empty()


@pytest.mark.asyncio
async def test_vad_parameters(transcriber, mock_model, audio_queue):
    """Test VAD parameters are passed correctly."""
    transcriber._model = mock_model
    transcriber.whisper_config.vad_filter = True
    transcriber.set_current_output_mode(CliOutputMode.KEYBOARD)

    await transcriber.start()
    await audio_queue.put(np.zeros(16000, dtype=np.float32))
    await asyncio.sleep(0.1)  # Give time for processing

    # Check VAD parameters were passed
    call = mock_model.transcribe.mock_calls[0]  # Get first call
    call_kwargs = call.kwargs
    assert call_kwargs["vad_filter"] is True
    assert "vad_parameters" in call_kwargs
    assert call_kwargs["vad_parameters"]["threshold"] == 0.5


@pytest.mark.asyncio
async def test_stop_clears_output_mode(transcriber, mock_model):
    """Test stopping clears output mode."""
    transcriber._model = mock_model
    transcriber.set_current_output_mode(CliOutputMode.KEYBOARD)

    await transcriber.start()
    assert transcriber._current_output_mode is not None

    await transcriber.stop()
    assert transcriber._current_output_mode is None


@pytest.mark.asyncio
async def test_multiple_start_stop(transcriber, mock_model):
    """Test multiple start/stop cycles."""
    transcriber._model = mock_model

    # First cycle
    await transcriber.start()
    task1 = transcriber._transcription_task
    await transcriber.stop()

    # Second cycle
    await transcriber.start()
    task2 = transcriber._transcription_task
    await transcriber.stop()

    assert task1 is not task2  # Should be different task objects
    assert transcriber._transcription_task is None  # Should be cleaned up
