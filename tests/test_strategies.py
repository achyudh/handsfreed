"""Tests for audio segmentation strategies."""

import asyncio
import pytest
import numpy as np
from unittest.mock import AsyncMock, Mock, patch

from handsfreed.audio_capture import SAMPLE_RATE
from handsfreed.ipc_models import CliOutputMode
from handsfreed.pipelines import TranscriptionTask
from handsfreed.strategies import TimeBasedSegmentationStrategy


@pytest.fixture
def config_mock():
    """Create a mock configuration object."""
    config = Mock()
    config.daemon = Mock()
    config.daemon.time_chunk_s = 0.5  # small value for testing
    return config


@pytest.fixture
def raw_audio_queue():
    """Create a queue for raw audio frames."""
    return asyncio.Queue()


@pytest.fixture
def transcription_queue():
    """Create a queue for transcription tasks."""
    return asyncio.Queue()


@pytest.fixture
def stop_event():
    """Create a stop event."""
    return asyncio.Event()


@pytest.fixture
def time_strategy(raw_audio_queue, transcription_queue, stop_event, config_mock):
    """Create a TimeBasedSegmentationStrategy instance."""
    return TimeBasedSegmentationStrategy(
        raw_audio_queue, transcription_queue, stop_event, config_mock
    )


@pytest.mark.asyncio
async def test_time_strategy_init(time_strategy, config_mock):
    """Test TimeBasedSegmentationStrategy initialization."""
    assert time_strategy.chunk_duration_s == config_mock.daemon.time_chunk_s
    assert time_strategy.chunk_size_frames == int(
        config_mock.daemon.time_chunk_s * SAMPLE_RATE
    )
    assert time_strategy._active_mode is None
    assert isinstance(time_strategy._buffer, np.ndarray)
    assert len(time_strategy._buffer) == 0


@pytest.mark.asyncio
async def test_time_strategy_set_active_mode(time_strategy):
    """Test setting active output mode."""
    # Initially None
    assert time_strategy._active_mode is None

    # Set to KEYBOARD
    await time_strategy.set_active_output_mode(CliOutputMode.KEYBOARD)
    assert time_strategy._active_mode == CliOutputMode.KEYBOARD

    # Set to CLIPBOARD
    await time_strategy.set_active_output_mode(CliOutputMode.CLIPBOARD)
    assert time_strategy._active_mode == CliOutputMode.CLIPBOARD

    # Set back to None (should clear buffer)
    time_strategy._buffer = np.ones(100, dtype=np.float32)
    await time_strategy.set_active_output_mode(None)
    assert time_strategy._active_mode is None
    assert len(time_strategy._buffer) == 0


@pytest.mark.asyncio
async def test_time_strategy_process_chunk(time_strategy):
    """Test time-based strategy produces chunks correctly."""
    # Set active mode
    await time_strategy.set_active_output_mode(CliOutputMode.KEYBOARD)

    # Create test audio frames (each 0.1s at SAMPLE_RATE)
    frame_size = int(0.1 * SAMPLE_RATE)
    frames = [np.ones(frame_size, dtype=np.float32) * i for i in range(1, 6)]

    # Start processing in background
    process_task = asyncio.create_task(time_strategy.process())

    try:
        # Put frames on the queue
        for frame in frames:
            await time_strategy.raw_audio_queue.put(frame)

        # Wait for processing (up to 1s)
        await asyncio.sleep(0.5)

        # Check we got a transcription task with the right audio
        # (0.5s chunk size with 5 * 0.1s frames = 1 complete chunk)
        assert not time_strategy.transcription_queue.empty()

        task = await time_strategy.transcription_queue.get()
        assert isinstance(task, TranscriptionTask)
        assert task.output_mode == CliOutputMode.KEYBOARD
        assert isinstance(task.audio, np.ndarray)
        assert len(task.audio) == time_strategy.chunk_size_frames

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

    finally:
        # Stop the background task
        process_task.cancel()
        await asyncio.gather(process_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_time_strategy_no_output_when_inactive(time_strategy):
    """Test time-based strategy doesn't produce output when inactive."""
    # Make sure active mode is None
    await time_strategy.set_active_output_mode(None)

    # Create test audio frames (each 0.1s at SAMPLE_RATE)
    frame_size = int(0.1 * SAMPLE_RATE)
    frames = [np.ones(frame_size, dtype=np.float32) * i for i in range(1, 10)]

    # Start processing in background
    process_task = asyncio.create_task(time_strategy.process())

    try:
        # Put frames on the queue (should produce 1.5 chunks worth of data)
        for frame in frames:
            await time_strategy.raw_audio_queue.put(frame)

        # Wait for processing (up to 0.5s)
        await asyncio.sleep(0.5)

        # Check no transcription tasks were produced
        assert time_strategy.transcription_queue.empty()

    finally:
        # Stop the background task
        process_task.cancel()
        await asyncio.gather(process_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_time_strategy_stop():
    """Test stopping time-based strategy processing."""
    # Need to patch the stop method as we can't mock coroutines easily in Python
    with patch("handsfreed.pipelines.SegmentationStrategy.stop") as mock_stop:
        # Create a mock queue
        raw_queue = asyncio.Queue()
        trans_queue = asyncio.Queue()
        stop_event = asyncio.Event()
        config = Mock()
        config.daemon = Mock()
        config.daemon.time_chunk_s = 0.5

        # Create the strategy
        strategy = TimeBasedSegmentationStrategy(
            raw_queue, trans_queue, stop_event, config
        )

        # Mock a task
        task = AsyncMock()
        task.done.return_value = False
        task.cancel = Mock()
        strategy._processing_task = task

        # Call stop
        await strategy.stop()

        # Since we're patching the parent class stop method,
        # verify it was called with the right args
        mock_stop.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_time_strategy_respects_stop_event(time_strategy, stop_event):
    """Test time strategy stops when stop event is set."""
    # Start processing in background
    process_task = asyncio.create_task(time_strategy.process())

    # Set active mode
    await time_strategy.set_active_output_mode(CliOutputMode.KEYBOARD)

    # Wait briefly
    await asyncio.sleep(0.1)

    # Set the stop event
    stop_event.set()

    # Wait for task completion
    try:
        await asyncio.wait_for(process_task, timeout=1)
    except asyncio.TimeoutError:
        pytest.fail("Strategy didn't respect stop event")

    # Task should be done
    assert process_task.done()
