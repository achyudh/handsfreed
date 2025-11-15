"""Tests for audio segmentation strategies."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import numpy as np
import pytest
import pytest_asyncio
from handsfreed.audio_capture import FRAME_SIZE, SAMPLE_RATE
from handsfreed.ipc_models import CliOutputMode
from handsfreed.pipeline import TranscriptionTask
from handsfreed.segmentation import (
    FixedSegmentationStrategy,
    VADSegmentationStrategy,
)


@pytest.fixture
def config_mock():
    """Create a mock configuration object."""
    config = Mock()
    config.daemon = Mock()
    config.daemon.time_chunk_s = 0.5  # small value for testing
    return config


@pytest_asyncio.fixture
async def raw_audio_queue():
    """Create a queue for raw audio frames."""
    queue = asyncio.Queue()
    yield queue
    while not queue.empty():
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            break


@pytest_asyncio.fixture
async def transcription_queue():
    """Create a queue for transcription tasks."""
    queue = asyncio.Queue()
    yield queue
    while not queue.empty():
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            break


@pytest.fixture
def stop_event():
    """Create a stop event."""
    return asyncio.Event()


@pytest_asyncio.fixture
async def fixed_strategy(raw_audio_queue, transcription_queue, stop_event, config_mock):
    """Create a FixedSegmentationStrategy instance."""
    strategy = FixedSegmentationStrategy(
        raw_audio_queue, transcription_queue, stop_event, config_mock
    )
    yield strategy
    # Clean up if needed
    await strategy.stop()


@pytest.fixture
def vad_config_mock():
    """Create a mock VAD configuration object."""
    config = Mock()
    config.vad = Mock()
    config.vad.enabled = True
    config.vad.threshold = 0.5
    config.vad.min_speech_duration_ms = 250
    config.vad.min_silence_duration_ms = 500
    config.vad.pre_roll_duration_ms = 200
    config.vad.neg_threshold = 0.3
    config.vad.max_speech_duration_s = 10.0
    return config


@pytest.fixture
def vad_model_mock():
    """Create a mock VAD model."""
    return Mock()


@pytest_asyncio.fixture
async def vad_strategy(
    raw_audio_queue, transcription_queue, stop_event, vad_config_mock, vad_model_mock
):
    """Create a VADSegmentationStrategy instance."""
    strategy = VADSegmentationStrategy(
        raw_audio_queue,
        transcription_queue,
        stop_event,
        vad_config_mock,
        vad_model_mock,
    )
    yield strategy
    # Clean up if needed
    await strategy.stop()


@pytest.mark.asyncio
async def test_fixed_strategy_init(fixed_strategy, config_mock):
    """Test FixedSegmentationStrategy initialization."""
    assert fixed_strategy.chunk_duration_s == config_mock.daemon.time_chunk_s
    assert fixed_strategy.chunk_size_frames == int(
        config_mock.daemon.time_chunk_s * SAMPLE_RATE
    )
    assert fixed_strategy._active_mode is None
    assert isinstance(fixed_strategy._buffer, np.ndarray)
    assert len(fixed_strategy._buffer) == 0


@pytest.mark.asyncio
async def test_fixed_strategy_set_active_mode(fixed_strategy):
    """Test setting active output mode."""
    # Initially None
    assert fixed_strategy._active_mode is None

    # Set to KEYBOARD
    await fixed_strategy.set_active_output_mode(CliOutputMode.KEYBOARD)
    assert fixed_strategy._active_mode == CliOutputMode.KEYBOARD

    # Set to CLIPBOARD
    await fixed_strategy.set_active_output_mode(CliOutputMode.CLIPBOARD)
    assert fixed_strategy._active_mode == CliOutputMode.CLIPBOARD

    # Set back to None (should clear buffer)
    fixed_strategy._buffer = np.ones(100, dtype=np.float32)
    await fixed_strategy.set_active_output_mode(None)
    assert fixed_strategy._active_mode is None
    assert len(fixed_strategy._buffer) == 0


@pytest.mark.asyncio
async def test_fixed_strategy_process_chunk(fixed_strategy):
    """Test fixed-duration strategy produces chunks correctly."""
    # Set active mode
    await fixed_strategy.set_active_output_mode(CliOutputMode.KEYBOARD)

    # Create test audio frames (each 0.1s at SAMPLE_RATE)
    frame_size = int(0.1 * SAMPLE_RATE)
    frames = [np.ones(frame_size, dtype=np.float32) * i for i in range(1, 6)]

    # Start processing
    await fixed_strategy.start()

    # Put frames on the queue
    for frame in frames:
        await fixed_strategy.input_queue.put(frame)
        # Let the loop process
        await asyncio.sleep(0.01)

    # Check we got a transcription task with the right audio
    # (0.5s chunk size with 5 * 0.1s frames = 1 complete chunk)
    assert not fixed_strategy.output_queue.empty()

    task = await fixed_strategy.output_queue.get()
    assert isinstance(task, TranscriptionTask)
    assert task.output_mode == CliOutputMode.KEYBOARD
    assert isinstance(task.audio, np.ndarray)
    assert len(task.audio) == fixed_strategy.chunk_size_frames

    # The task should contain the first 0.5s of audio (first 5 frames)
    assert np.array_equal(task.audio[:frame_size], np.ones(frame_size) * 1)
    assert np.array_equal(
        task.audio[frame_size : frame_size * 2], np.ones(frame_size) * 2
    )
    assert np.array_equal(
        task.audio[frame_size * 2 : frame_size * 3], np.ones(frame_size) * 3
    )
    assert np.array_equal(
        task.audio[frame_size * 3 : frame_size * 4], np.ones(frame_size) * 4
    )
    assert np.array_equal(
        task.audio[frame_size * 4 : frame_size * 5], np.ones(frame_size) * 5
    )


@pytest.mark.asyncio
async def test_fixed_strategy_no_output_when_inactive(fixed_strategy):
    """Test fixed-duration strategy doesn't produce output when inactive."""
    # Make sure active mode is None
    await fixed_strategy.set_active_output_mode(None)

    # Create test audio frames (each 0.1s at SAMPLE_RATE)
    frame_size = int(0.1 * SAMPLE_RATE)
    frames = [np.ones(frame_size, dtype=np.float32) * i for i in range(1, 10)]

    # Start processing
    await fixed_strategy.start()

    # Put frames on the queue (should produce 1.5 chunks worth of data)
    for frame in frames:
        await fixed_strategy.input_queue.put(frame)
        await asyncio.sleep(0.01)

    # Check no transcription tasks were produced
    assert fixed_strategy.output_queue.empty()


@pytest.mark.asyncio
async def test_fixed_strategy_stop():
    """Test stopping fixed-duration strategy processing."""
    # Create a mock queue
    raw_queue = asyncio.Queue()
    trans_queue = asyncio.Queue()
    stop_event = asyncio.Event()
    config = Mock()
    config.daemon = Mock()
    config.daemon.time_chunk_s = 0.5

    # Create the strategy
    strategy = FixedSegmentationStrategy(raw_queue, trans_queue, stop_event, config)

    # Start the strategy
    await strategy.start()

    # Stop the strategy
    await strategy.stop()

    # Check that the processing task is no longer running
    assert strategy._task is None or strategy._task.done()


@pytest.mark.asyncio
async def test_fixed_strategy_respects_stop_event(fixed_strategy, stop_event):
    """Test fixed strategy stops when stop event is set."""
    # Start processing
    await fixed_strategy.start()

    # Set active mode
    await fixed_strategy.set_active_output_mode(CliOutputMode.KEYBOARD)

    # Wait briefly
    await asyncio.sleep(0.01)

    # Set the stop event
    stop_event.set()

    # Wait for task completion
    try:
        await asyncio.wait_for(fixed_strategy._task, timeout=1)
    except asyncio.TimeoutError:
        pytest.fail("Strategy didn't respect stop event")

    # Task should be done
    assert fixed_strategy._task.done()
