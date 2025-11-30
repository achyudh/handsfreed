"""Tests for audio segmentation strategies."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import numpy as np
import pytest
import pytest_asyncio
from handsfreed.audio_capture import FRAME_SIZE, SAMPLE_RATE, AUDIO_DTYPE
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
async def segment_queue():
    """Create a queue for audio segments."""
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
async def fixed_strategy(raw_audio_queue, segment_queue, stop_event, config_mock):
    """Create a FixedSegmentationStrategy instance."""
    strategy = FixedSegmentationStrategy(
        raw_audio_queue, segment_queue, stop_event, config_mock
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
    raw_audio_queue, segment_queue, stop_event, vad_config_mock, vad_model_mock
):
    """Create a VADSegmentationStrategy instance."""
    strategy = VADSegmentationStrategy(
        raw_audio_queue,
        segment_queue,
        stop_event,
        vad_config_mock,
        vad_model_mock,
    )
    yield strategy
    # Clean up if needed
    await strategy.stop()


@pytest.mark.asyncio
async def test_vad_strategy_set_enabled_clears_buffers(vad_strategy):
    """Test that disabling VAD strategy clears internal buffers."""
    # Populate buffers
    vad_strategy._pre_roll_buffer.append(np.zeros(10, dtype=AUDIO_DTYPE))
    vad_strategy._current_segment.append(np.zeros(10, dtype=AUDIO_DTYPE))

    # Enable first (to set _enabled=True)
    await vad_strategy.set_enabled(True)

    # Disable
    await vad_strategy.set_enabled(False)

    # Verify buffers are empty
    assert len(vad_strategy._pre_roll_buffer) == 0
    assert len(vad_strategy._current_segment) == 0


@pytest.mark.asyncio
async def test_fixed_strategy_init(fixed_strategy, config_mock):
    """Test FixedSegmentationStrategy initialization."""
    assert fixed_strategy.chunk_duration_s == config_mock.daemon.time_chunk_s
    assert fixed_strategy.chunk_size_frames == int(
        config_mock.daemon.time_chunk_s * SAMPLE_RATE
    )
    assert fixed_strategy._enabled is False
    assert isinstance(fixed_strategy._buffer, np.ndarray)
    assert len(fixed_strategy._buffer) == 0


@pytest.mark.asyncio
async def test_fixed_strategy_set_enabled(fixed_strategy):
    """Test setting enabled state."""
    # Initially False
    assert fixed_strategy._enabled is False

    # Set to True
    await fixed_strategy.set_enabled(True)
    assert fixed_strategy._enabled is True

    # Set back to False (should clear buffer)
    fixed_strategy._buffer = np.ones(100, dtype=np.float32)
    await fixed_strategy.set_enabled(False)
    assert fixed_strategy._enabled is False
    assert len(fixed_strategy._buffer) == 0


@pytest.mark.asyncio
async def test_fixed_strategy_process_chunk(fixed_strategy):
    """Test fixed-duration strategy produces chunks correctly."""
    # Set enabled
    await fixed_strategy.set_enabled(True)

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

    # Check we got a segment with the right audio
    # (0.5s chunk size with 5 * 0.1s frames = 1 complete chunk)
    assert not fixed_strategy.output_queue.empty()

    segment = await fixed_strategy.output_queue.get()
    assert isinstance(segment, np.ndarray)
    assert len(segment) == fixed_strategy.chunk_size_frames

    # The segment should contain the first 0.5s of audio (first 5 frames)
    assert np.array_equal(segment[:frame_size], np.ones(frame_size) * 1)
    assert np.array_equal(segment[frame_size : frame_size * 2], np.ones(frame_size) * 2)
    assert np.array_equal(
        segment[frame_size * 2 : frame_size * 3], np.ones(frame_size) * 3
    )
    assert np.array_equal(
        segment[frame_size * 3 : frame_size * 4], np.ones(frame_size) * 4
    )
    assert np.array_equal(
        segment[frame_size * 4 : frame_size * 5], np.ones(frame_size) * 5
    )


@pytest.mark.asyncio
async def test_fixed_strategy_no_output_when_inactive(fixed_strategy):
    """Test fixed-duration strategy doesn't produce output when inactive."""
    # Make sure disabled
    await fixed_strategy.set_enabled(False)

    # Create test audio frames (each 0.1s at SAMPLE_RATE)
    frame_size = int(0.1 * SAMPLE_RATE)
    frames = [np.ones(frame_size, dtype=np.float32) * i for i in range(1, 10)]

    # Start processing
    await fixed_strategy.start()

    # Put frames on the queue (should produce 1.5 chunks worth of data)
    for frame in frames:
        await fixed_strategy.input_queue.put(frame)
        await asyncio.sleep(0.01)

    # Check no segments were produced
    assert fixed_strategy.output_queue.empty()


@pytest.mark.asyncio
async def test_fixed_strategy_stop():
    """Test stopping fixed-duration strategy processing."""
    # Create a mock queue
    raw_queue = asyncio.Queue()
    seg_queue = asyncio.Queue()
    stop_event = asyncio.Event()
    config = Mock()
    config.daemon = Mock()
    config.daemon.time_chunk_s = 0.5

    # Create the strategy
    strategy = FixedSegmentationStrategy(raw_queue, seg_queue, stop_event, config)

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

    # Set enabled
    await fixed_strategy.set_enabled(True)

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
