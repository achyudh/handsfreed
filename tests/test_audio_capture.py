"""Tests for audio capture module."""

import asyncio
import pytest
import pytest_asyncio
import numpy as np
import sounddevice as sd
from unittest.mock import MagicMock, patch

from handsfreed.audio_capture import AudioCapture, AUDIO_DTYPE, FRAME_SIZE
from handsfreed.config import AudioConfig


@pytest.fixture
def raw_audio_queue():
    """Create a raw audio queue."""
    return asyncio.Queue()


@pytest.fixture
def mock_stream():
    """Mock sounddevice.InputStream."""
    stream = MagicMock(spec=sd.InputStream)
    stream.start = MagicMock()
    stream.stop = MagicMock()
    stream.close = MagicMock()
    return stream


@pytest.fixture
def stop_event():
    """Create a stop event."""
    return asyncio.Event()


@pytest_asyncio.fixture
async def audio_capture(raw_audio_queue, stop_event):
    """Create an audio capture instance."""
    audio_config = AudioConfig()
    capture = AudioCapture(raw_audio_queue, audio_config, stop_event)
    yield capture
    # Always call stop to ensure the processing task is cleaned up.
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
        assert audio_capture._task is not None
        mock_stream.start.assert_called_once()


@pytest.mark.asyncio
async def test_stop_success(audio_capture, mock_stream):
    """Test successful stop of audio capture."""
    with patch("sounddevice.InputStream", return_value=mock_stream):
        await audio_capture.start()
        await audio_capture.stop()

        assert audio_capture._stream is None
        assert audio_capture._task is None
        mock_stream.stop.assert_called_once()
        mock_stream.close.assert_called_once()


@pytest.mark.asyncio
async def test_raw_audio_processing(audio_capture, mock_stream, raw_audio_queue):
    """Test raw audio data is correctly processed."""
    with patch("sounddevice.InputStream", return_value=mock_stream):
        await audio_capture.start()

        # Feed frame-size of data
        test_data = np.ones(FRAME_SIZE, dtype=AUDIO_DTYPE)
        await simulate_audio_data(audio_capture, test_data)

        # Should get the frame on the queue
        frame = await asyncio.wait_for(raw_audio_queue.get(), timeout=1.0)
        assert isinstance(frame, np.ndarray)
        assert len(frame) == FRAME_SIZE
        assert frame.dtype == AUDIO_DTYPE

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
        assert audio_capture._task is None


@pytest.mark.asyncio
async def test_gain_control(audio_capture, mock_stream, raw_audio_queue):
    """Test input gain control."""
    audio_capture.audio_config.input_gain = 2.0  # Double the input
    # Disable DC offset for this specific test to isolate gain functionality
    audio_capture.audio_config.dc_offset_correction = False

    with patch("sounddevice.InputStream", return_value=mock_stream):
        await audio_capture.start()

        # Create a sine wave with amplitude 0.5
        t = np.linspace(0, FRAME_SIZE / 16000, FRAME_SIZE, endpoint=False)
        test_data = 0.5 * np.sin(2 * np.pi * 440 * t).astype(AUDIO_DTYPE)

        await simulate_audio_data(audio_capture, test_data)

        # Get processed frame
        frame = await asyncio.wait_for(raw_audio_queue.get(), timeout=1.0)

        # The peak of the sine wave should be doubled from 0.5 to 1.0
        assert np.allclose(np.max(frame), 1.0, atol=1e-6)

        await audio_capture.stop()


@pytest.mark.asyncio
async def test_dc_offset_correction(audio_capture, mock_stream, raw_audio_queue):
    """Test DC offset correction."""
    audio_capture.audio_config.dc_offset_correction = True

    with patch("sounddevice.InputStream", return_value=mock_stream):
        await audio_capture.start()

        # Create test data with a DC offset of 0.2
        test_data = np.ones(FRAME_SIZE, dtype=AUDIO_DTYPE) * 0.5 + 0.2
        await simulate_audio_data(audio_capture, test_data)

        # Get processed frame
        frame = await asyncio.wait_for(raw_audio_queue.get(), timeout=1.0)

        # The running average will not be perfect on the first frame, but it should be close
        # For a single frame, the offset removed will be the mean of that frame.
        assert np.allclose(frame.mean(), 0, atol=1e-6)

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
