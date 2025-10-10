"""Tests for audio capture module using pw-record."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import pytest_asyncio

from handsfreed.audio_capture import PWRecordCapture
from handsfreed.ipc_models import CliOutputMode
from handsfreed.pipelines import TranscriptionTask


@pytest.fixture
def config_mock():
    """Create a mock configuration object for audio settings."""
    config = MagicMock()
    config.audio.target = "test-target"
    return config


@pytest_asyncio.fixture
async def transcription_queue():
    """Create a queue for transcription tasks."""
    return asyncio.Queue()


@pytest_asyncio.fixture
async def audio_capture(config_mock, transcription_queue):
    """Create a PWRecordCapture instance."""
    capture = PWRecordCapture(config_mock, transcription_queue)
    yield capture
    # Ensure cleanup if a task is running
    if capture._is_running:
        await capture.stop()


@pytest.fixture
def mock_subprocess():
    """Create a mock asyncio.subprocess.Process."""
    process = AsyncMock()
    process.pid = 1234
    process.returncode = None

    # terminate() is a sync method
    process.terminate = MagicMock()

    # Mock stdout stream reader
    process.stdout = AsyncMock(spec=asyncio.StreamReader)
    # Simulate some audio data then EOF
    fake_audio_chunk = np.array([1000, -1000, 2000, -2000], dtype=np.int16).tobytes()
    process.stdout.read.side_effect = [fake_audio_chunk, b""]  # Send data then EOF

    return process


@pytest.mark.asyncio
@patch("asyncio.create_subprocess_exec")
async def test_start_success(mock_create_subprocess, audio_capture, mock_subprocess):
    """Test that start() correctly initiates the pw-record process."""
    mock_create_subprocess.return_value = mock_subprocess

    await audio_capture.start(CliOutputMode.KEYBOARD)

    # Check that subprocess was called correctly
    mock_create_subprocess.assert_called_once_with(
        "pw-record",
        "--target=test-target",
        "--rate=16000",
        "--format=s16",
        "--channels=1",
        "-",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )

    assert audio_capture._is_running is True
    assert audio_capture._process is mock_subprocess
    assert audio_capture._reader_task is not None
    assert audio_capture._output_mode == CliOutputMode.KEYBOARD

    # Clean up the task to avoid warnings
    if audio_capture._reader_task:
        audio_capture._reader_task.cancel()
        try:
            await audio_capture._reader_task
        except asyncio.CancelledError:
            pass


@pytest.mark.asyncio
@patch("asyncio.create_subprocess_exec")
async def test_stop_and_transcribe(mock_create_subprocess, audio_capture, mock_subprocess, transcription_queue):
    """Test that stop() terminates the process and queues a transcription task."""
    mock_create_subprocess.return_value = mock_subprocess

    # Start capture
    await audio_capture.start(CliOutputMode.KEYBOARD)

    # Let the reader task run to populate the buffer
    await asyncio.sleep(0.01)

    # Stop capture
    await audio_capture.stop()

    # Assertions
    assert audio_capture._is_running is False
    mock_subprocess.terminate.assert_called_once()

    # Check that a task was put on the transcription queue
    assert not transcription_queue.empty()
    task = await transcription_queue.get()
    assert isinstance(task, TranscriptionTask)
    assert task.output_mode == CliOutputMode.KEYBOARD

    # Verify the audio data is correct
    int_array = np.array([1000, -1000, 2000, -2000], dtype=np.int16)
    expected_array = int_array.astype(np.float32) * (1.0 / 32768.0)
    np.testing.assert_allclose(task.audio, expected_array, atol=1e-5)


@pytest.mark.asyncio
@patch("asyncio.create_subprocess_exec")
async def test_start_while_running(mock_create_subprocess, audio_capture, mock_subprocess):
    """Test that calling start() while already running is a no-op."""
    mock_create_subprocess.return_value = mock_subprocess

    await audio_capture.start(CliOutputMode.KEYBOARD)
    # Check it was called the first time
    mock_create_subprocess.assert_called_once()

    # Try to start again
    await audio_capture.start(CliOutputMode.CLIPBOARD)
    # Should not have been called again
    mock_create_subprocess.assert_called_once()


@pytest.mark.asyncio
async def test_stop_when_not_running(audio_capture):
    """Test that calling stop() when not running is a no-op."""
    # Patch the queue to ensure no task is added
    with patch.object(audio_capture.transcription_queue, "put") as mock_put:
        await audio_capture.stop()
        mock_put.assert_not_called()


@pytest.mark.asyncio
@patch("asyncio.create_subprocess_exec", side_effect=FileNotFoundError("pw-record not found"))
async def test_pw_record_not_found(mock_create_subprocess, audio_capture):
    """Test that a FileNotFoundError is raised if pw-record is not found."""
    with pytest.raises(FileNotFoundError, match="pw-record not found"):
        await audio_capture.start(CliOutputMode.KEYBOARD)

    assert audio_capture._is_running is False