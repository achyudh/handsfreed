"""Tests for audio capture module."""

import asyncio
import pytest
import pytest_asyncio
import numpy as np
import sounddevice as sd
from unittest.mock import MagicMock, patch, AsyncMock

from handsfreed.audio_capture import AudioCapture, SAMPLE_RATE, AUDIO_DTYPE


@pytest.fixture
def audio_queue():
    """Create an audio queue."""
    return asyncio.Queue()


@pytest.fixture
def mock_stream():
    """Mock sounddevice.InputStream."""
    stream = MagicMock(spec=sd.InputStream)
    stream.start = MagicMock()
    stream.stop = MagicMock()
    stream.close = MagicMock()
    return stream


@pytest_asyncio.fixture
async def audio_capture(audio_queue):
    """Create an audio capture instance."""
    capture = AudioCapture(
        audio_queue, chunk_duration_s=0.1
    )  # Short chunks for testing
    yield capture
    # Cleanup if needed
    if capture._stream is not None:
        await capture.stop()


async def simulate_audio_data(capture, data: np.ndarray) -> None:
    """Helper to simulate audio data and wait for processing."""
    # Reshape to mono if needed
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)

    # Send via callback
    capture._audio_callback(data, len(data), None, None)

    # Give processor time to handle data
    await asyncio.sleep(0.1)


@pytest.mark.asyncio
async def test_start_success(audio_capture, mock_stream):
    """Test successful start of audio capture."""
    with patch("sounddevice.InputStream", return_value=mock_stream):
        await audio_capture.start()

        assert audio_capture._stream is not None
        assert audio_capture._processing_task is not None
        mock_stream.start.assert_called_once()


@pytest.mark.asyncio
async def test_stop_success(audio_capture, mock_stream):
    """Test successful stop of audio capture."""
    with patch("sounddevice.InputStream", return_value=mock_stream):
        await audio_capture.start()
        await audio_capture.stop()

        assert audio_capture._stream is None
        assert audio_capture._processing_task is None
        mock_stream.stop.assert_called_once()
        mock_stream.close.assert_called_once()


@pytest.mark.asyncio
async def test_audio_chunking(audio_capture, mock_stream, audio_queue):
    """Test audio data is correctly chunked."""
    chunk_duration = 0.1  # seconds
    chunk_size = int(SAMPLE_RATE * chunk_duration)
    audio_capture.chunk_duration_s = chunk_duration

    with patch("sounddevice.InputStream", return_value=mock_stream):
        await audio_capture.start()

        # Feed 2.5 chunks worth of data
        test_data = np.ones(int(chunk_size * 2.5), dtype=AUDIO_DTYPE)
        await simulate_audio_data(audio_capture, test_data)

        # Should get 2 full chunks
        chunk1 = audio_queue._queue[0]  # Direct access for testing
        assert isinstance(chunk1, np.ndarray)
        assert len(chunk1) == chunk_size
        assert chunk1.dtype == AUDIO_DTYPE

        chunk2 = audio_queue._queue[1]
        assert isinstance(chunk2, np.ndarray)
        assert len(chunk2) == chunk_size

        # Partial chunk should remain in buffer
        assert len(audio_capture._buffer) == int(chunk_size * 0.5)

        await audio_capture.stop()


@pytest.mark.asyncio
async def test_stream_error_handling(audio_capture):
    """Test handling of PortAudio errors."""
    mock_stream = MagicMock(spec=sd.InputStream)
    mock_stream.start.side_effect = sd.PortAudioError("Test error")

    with patch("sounddevice.InputStream", return_value=mock_stream):
        with pytest.raises(sd.PortAudioError, match="Test error"):
            await audio_capture.start()

        assert audio_capture._stream is None
        assert audio_capture._processing_task is None


@pytest.mark.asyncio
async def test_gain_control(audio_capture, mock_stream, audio_queue):
    """Test input gain control."""
    audio_capture.input_gain = 2.0  # Double the input

    with patch("sounddevice.InputStream", return_value=mock_stream):
        await audio_capture.start()

        # Create test data (0.5 amplitude)
        test_data = np.ones(SAMPLE_RATE // 10, dtype=AUDIO_DTYPE) * 0.5
        await simulate_audio_data(audio_capture, test_data)

        # Get processed chunk
        chunk = audio_queue._queue[0]  # Direct access for testing
        assert np.allclose(chunk, 1.0)  # Should be doubled from 0.5 to 1.0

        await audio_capture.stop()


@pytest.mark.asyncio
async def test_multiple_start_stop(audio_capture, mock_stream):
    """Test multiple start/stop cycles."""
    with patch("sounddevice.InputStream", return_value=mock_stream):
        # First cycle
        await audio_capture.start()
        assert audio_capture._stream is not None
        await audio_capture.stop()
        assert audio_capture._stream is None

        # Second cycle
        await audio_capture.start()
        assert audio_capture._stream is not None
        await audio_capture.stop()
        assert audio_capture._stream is None

        assert mock_stream.start.call_count == 2
        assert mock_stream.stop.call_count == 2
