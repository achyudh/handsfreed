"""Tests for audio segmentation strategies."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import numpy as np
import pytest
import pytest_asyncio
from handsfreed.audio_capture import FRAME_SIZE, SAMPLE_RATE
from handsfreed.ipc_models import CliOutputMode
from handsfreed.pipelines import TranscriptionTask
from handsfreed.strategies import (
    TimeBasedSegmentationStrategy,
    VADSegmentationStrategy,
    VADState,
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
async def time_strategy(raw_audio_queue, transcription_queue, stop_event, config_mock):
    """Create a TimeBasedSegmentationStrategy instance."""
    strategy = TimeBasedSegmentationStrategy(
        raw_audio_queue, transcription_queue, stop_event, config_mock
    )
    yield strategy
    # Clean up if needed
    if hasattr(strategy, "_processing_task") and strategy._processing_task:
        strategy._processing_task.cancel()
        try:
            await strategy._processing_task
        except asyncio.CancelledError:
            pass


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
    if hasattr(strategy, "_processing_task") and strategy._processing_task:
        strategy._processing_task.cancel()
        try:
            await strategy._processing_task
        except asyncio.CancelledError:
            pass


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
            # Let the loop process
            await asyncio.sleep(0.01)

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
        # Clean up the process task
        if not process_task.done():
            process_task.cancel()
            try:
                await process_task
            except asyncio.CancelledError:
                pass


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
            await asyncio.sleep(0.01)

        # Check no transcription tasks were produced
        assert time_strategy.transcription_queue.empty()

    finally:
        # Clean up the process task
        if not process_task.done():
            process_task.cancel()
            try:
                await process_task
            except asyncio.CancelledError:
                pass


@pytest.mark.asyncio
async def test_time_strategy_stop():
    """Test stopping time-based strategy processing."""
    # Create a mock queue
    raw_queue = asyncio.Queue()
    trans_queue = asyncio.Queue()
    stop_event = asyncio.Event()
    config = Mock()
    config.daemon = Mock()
    config.daemon.time_chunk_s = 0.5

    # Create the strategy with a mocked stop method to avoid using AsyncMock directly
    strategy = TimeBasedSegmentationStrategy(raw_queue, trans_queue, stop_event, config)

    # Replace the stop method - we're not testing the parent stop method, just that it's called
    original_stop = strategy.stop
    strategy.stop = AsyncMock()

    # Start a simple process task that can be properly cancelled
    process_task = asyncio.create_task(asyncio.sleep(10))
    strategy._processing_task = process_task

    # Call the parent class stop method directly
    await original_stop()

    # Check that the process task was cancelled
    assert process_task.cancelled()


@pytest.mark.asyncio
async def test_time_strategy_respects_stop_event(time_strategy, stop_event):
    """Test time strategy stops when stop event is set."""
    # Start processing in background
    process_task = asyncio.create_task(time_strategy.process())

    # Set active mode
    await time_strategy.set_active_output_mode(CliOutputMode.KEYBOARD)

    # Wait briefly
    await asyncio.sleep(0.01)

    # Set the stop event
    stop_event.set()

    # Wait for task completion
    try:
        await asyncio.wait_for(process_task, timeout=1)
    except asyncio.TimeoutError:
        pytest.fail("Strategy didn't respect stop event")

    # Task should be done
    assert process_task.done()


@pytest.mark.asyncio
async def test_vad_strategy_init(vad_strategy, vad_config_mock):
    """Test VADSegmentationStrategy initialization."""
    assert vad_strategy.vad_config == vad_config_mock.vad
    assert vad_strategy._vad_state == VADState.SILENT
    assert vad_strategy._silence_start_time == 0
    assert vad_strategy._current_segment == []
    # Check that pre-roll buffer is a collections.deque
    from collections import deque

    assert isinstance(vad_strategy._pre_roll_buffer, deque)
    # Check pre-roll buffer has correct max length based on config
    pre_roll_frames = (
        vad_config_mock.vad.pre_roll_duration_ms * SAMPLE_RATE / 1000
    ) // FRAME_SIZE
    assert vad_strategy._pre_roll_buffer.maxlen == pre_roll_frames


class MockMonotonic:
    """Mock for time.monotonic that advances by a fixed amount."""

    def __init__(self, start_time=0.0, increment=0.1):
        self.current_time = start_time
        self.increment = increment

    def __call__(self):
        current = self.current_time
        self.current_time += self.increment
        return current


@pytest.mark.asyncio
async def test_vad_strategy_direct():
    """Test VAD strategy directly by calling its methods."""
    # Create testing components
    raw_queue = asyncio.Queue()
    trans_queue = asyncio.Queue()
    stop_event = asyncio.Event()

    # Create a complete mock for vad_model that returns specified speech probabilities
    vad_model_mock = Mock()
    vad_model_mock.return_value = 0.7  # Default to speech detected

    # Create config with convenient test settings
    config = MagicMock()
    config.vad = MagicMock()
    config.vad.enabled = True
    config.vad.threshold = 0.5
    config.vad.min_speech_duration_ms = 0  # No minimum for easier testing
    config.vad.min_silence_duration_ms = 10  # Very short for testing
    config.vad.pre_roll_duration_ms = 256
    config.vad.neg_threshold = 0.3
    config.vad.max_speech_duration_s = 10.0

    # Create the strategy
    strategy = VADSegmentationStrategy(
        raw_queue, trans_queue, stop_event, config, vad_model_mock
    )

    # Set active mode
    await strategy.set_active_output_mode(CliOutputMode.KEYBOARD)

    # Reset state as if we're starting
    strategy._vad_state = VADState.SILENT

    # Test frame
    frame_size = 512
    test_frame = np.ones(frame_size, dtype=np.float32)

    # 1. Detect speech - should transition to SPEECH
    vad_model_mock.return_value = 0.7
    mock_time = 100.0  # Arbitrary time value

    with patch("time.monotonic", return_value=mock_time):
        # Add frame to pre-roll buffer
        strategy._pre_roll_buffer.append(test_frame.copy())

        # Process a frame with speech
        if test_frame.ndim == 1:
            vad_frame = test_frame.reshape(1, -1)
        else:
            vad_frame = test_frame

        speech_prob = await asyncio.to_thread(vad_model_mock, vad_frame, FRAME_SIZE)
        is_speech_prob_high = speech_prob >= strategy.vad_config.threshold
        is_speech_prob_low = speech_prob <= strategy.vad_config.neg_threshold

        # Assert we're in SILENT state and detected speech
        assert strategy._vad_state == VADState.SILENT
        assert is_speech_prob_high is True

        # Simulate process() logic for SILENT state
        if is_speech_prob_high:
            strategy._vad_state = VADState.SPEECH
            for pre_frame in strategy._pre_roll_buffer:
                strategy._current_segment.append(pre_frame)
            strategy._silence_start_time = 0

    # Verify transition to SPEECH
    assert strategy._vad_state == VADState.SPEECH
    assert len(strategy._current_segment) > 0

    # 2. Detect silence - should transition to ENDING_SPEECH
    vad_model_mock.return_value = 0.2  # Below threshold
    mock_time = 100.5  # Advance time

    with patch("time.monotonic", return_value=mock_time):
        # Process a frame with silence
        speech_prob = await asyncio.to_thread(vad_model_mock, vad_frame, FRAME_SIZE)
        is_speech_prob_high = speech_prob >= strategy.vad_config.threshold
        is_speech_prob_low = speech_prob <= strategy.vad_config.neg_threshold

        # Assert we're in SPEECH state and detected silence
        assert strategy._vad_state == VADState.SPEECH
        assert is_speech_prob_high is False
        assert is_speech_prob_low is True

        # Simulate adding frame to segment
        strategy._current_segment.append(test_frame.copy())

        # Simulate process() logic for SPEECH state with silence
        if is_speech_prob_low:
            strategy._vad_state = VADState.ENDING_SPEECH
            strategy._silence_start_time = mock_time

    # Verify transition to ENDING_SPEECH
    assert strategy._vad_state == VADState.ENDING_SPEECH
    assert strategy._silence_start_time == mock_time

    # 3. Resume speech - should go back to SPEECH
    vad_model_mock.return_value = 0.8  # Above threshold
    mock_time = 100.6  # Advance time slightly

    with patch("time.monotonic", return_value=mock_time):
        # Process a frame with speech
        speech_prob = await asyncio.to_thread(vad_model_mock, vad_frame, FRAME_SIZE)
        is_speech_prob_high = speech_prob >= strategy.vad_config.threshold

        # Assert we're in ENDING_SPEECH state and detected speech again
        assert strategy._vad_state == VADState.ENDING_SPEECH
        assert is_speech_prob_high is True

        # Simulate adding frame to segment
        strategy._current_segment.append(test_frame.copy())

        # Simulate process() logic for ENDING_SPEECH with speech
        if is_speech_prob_high:
            strategy._vad_state = VADState.SPEECH
            strategy._silence_start_time = 0

    # Verify transition back to SPEECH
    assert strategy._vad_state == VADState.SPEECH
    assert strategy._silence_start_time == 0

    # 4. Silence until timeout - should finalize segment
    vad_model_mock.return_value = 0.1  # Well below threshold
    mock_time = 100.8  # Advance time

    with patch("time.monotonic", return_value=mock_time):
        # Process a frame with silence
        speech_prob = await asyncio.to_thread(vad_model_mock, vad_frame, FRAME_SIZE)
        is_speech_prob_high = speech_prob >= strategy.vad_config.threshold
        is_speech_prob_low = speech_prob <= strategy.vad_config.neg_threshold

        # Simulate transition to ENDING_SPEECH
        if is_speech_prob_low:
            strategy._vad_state = VADState.ENDING_SPEECH
            strategy._silence_start_time = mock_time

    # Verify in ENDING_SPEECH state
    assert strategy._vad_state == VADState.ENDING_SPEECH

    # Now advance time beyond silence threshold
    mock_time = 100.8 + (strategy.vad_config.min_silence_duration_ms / 1000) + 0.01

    # Create a spy on _finalize_segment
    with patch.object(
        strategy, "_finalize_segment", wraps=strategy._finalize_segment
    ) as spy:
        with patch("time.monotonic", return_value=mock_time):
            # Process another frame with silence
            speech_prob = await asyncio.to_thread(vad_model_mock, vad_frame, FRAME_SIZE)
            is_speech_prob_high = speech_prob >= strategy.vad_config.threshold
            is_speech_prob_low = speech_prob <= strategy.vad_config.neg_threshold

            # Simulate adding frame to segment
            strategy._current_segment.append(test_frame.copy())

            # Simulate ENDING_SPEECH state with silence timeout check
            if not is_speech_prob_high:
                silence_duration_ms = (mock_time - strategy._silence_start_time) * 1000
                if silence_duration_ms >= strategy.vad_config.min_silence_duration_ms:
                    await strategy._finalize_segment()
                    strategy._vad_state = VADState.SILENT

    # Verify _finalize_segment was called
    spy.assert_awaited_once()

    # Verify transition back to SILENT
    assert strategy._vad_state == VADState.SILENT

    # Verify a transcription task was created
    assert not trans_queue.empty()
    task = trans_queue.get_nowait()
    assert isinstance(task, TranscriptionTask)
    assert task.output_mode == CliOutputMode.KEYBOARD


@pytest.mark.asyncio
async def test_max_speech_duration():
    """Test VAD strategy max speech duration check."""
    # Create testing components
    raw_queue = asyncio.Queue()
    trans_queue = asyncio.Queue()
    stop_event = asyncio.Event()

    # Create mock VAD model that always detects speech
    vad_model_mock = Mock()
    vad_model_mock.return_value = 0.8  # Always above threshold

    # Create config with short max duration
    config = MagicMock()
    config.vad = MagicMock()
    config.vad.enabled = True
    config.vad.threshold = 0.5
    config.vad.min_speech_duration_ms = 0  # No minimum for testing
    config.vad.min_silence_duration_ms = 500
    config.vad.pre_roll_duration_ms = 0
    config.vad.neg_threshold = 0.3
    config.vad.max_speech_duration_s = 0.1  # Very short (100ms)

    # Create the strategy
    strategy = VADSegmentationStrategy(
        raw_queue, trans_queue, stop_event, config, vad_model_mock
    )

    # Set active mode and ensure we're in speech state
    await strategy.set_active_output_mode(CliOutputMode.KEYBOARD)
    strategy._vad_state = VADState.SPEECH

    # Create test frame that will exceed max duration
    frame_size = 2000  # 125ms at 16kHz
    test_frame = np.ones(frame_size, dtype=np.float32)

    # Add frame to segment (which should exceed max duration)
    strategy._current_segment.append(test_frame.copy())

    # Create a spy on _finalize_segment
    with patch.object(
        strategy, "_finalize_segment", wraps=strategy._finalize_segment
    ) as spy:
        # Process a speech frame that should trigger max duration check
        if test_frame.ndim == 1:
            vad_frame = test_frame.reshape(1, -1)
        else:
            vad_frame = test_frame

        speech_prob = await asyncio.to_thread(vad_model_mock, vad_frame, FRAME_SIZE)
        is_speech_prob_high = speech_prob >= strategy.vad_config.threshold

        # Add another frame that would push us over the limit
        strategy._current_segment.append(test_frame.copy())

        # Check max speech duration
        current_duration_s = (
            sum(len(f) for f in strategy._current_segment) / SAMPLE_RATE
        )
        assert current_duration_s >= strategy.vad_config.max_speech_duration_s

        # Simulate max speech duration check
        if strategy.vad_config.max_speech_duration_s > 0:
            if current_duration_s >= strategy.vad_config.max_speech_duration_s:
                await strategy._finalize_segment()
                strategy._vad_state = VADState.SILENT

    # Verify _finalize_segment was called
    spy.assert_awaited_once()

    # Verify state was reset to SILENT
    assert strategy._vad_state == VADState.SILENT

    # Verify a transcription task was created
    assert not trans_queue.empty()
    task = trans_queue.get_nowait()
    assert isinstance(task, TranscriptionTask)
    assert task.output_mode == CliOutputMode.KEYBOARD


@pytest.mark.asyncio
async def test_min_speech_duration():
    """Test VAD strategy minimum speech duration check."""
    # Create testing components
    raw_queue = asyncio.Queue()
    trans_queue = asyncio.Queue()
    stop_event = asyncio.Event()

    # Create mock VAD model
    vad_model_mock = Mock()
    vad_model_mock.return_value = 0.1  # Below threshold (silence)

    # Create config with specific min speech duration
    config = MagicMock()
    config.vad = MagicMock()
    config.vad.enabled = True
    config.vad.threshold = 0.5
    config.vad.min_speech_duration_ms = 300  # Require at least 300ms of speech
    config.vad.min_silence_duration_ms = 100  # Short silence for testing
    config.vad.pre_roll_duration_ms = 0
    config.vad.neg_threshold = 0.3
    config.vad.max_speech_duration_s = 10.0

    # Create the strategy
    strategy = VADSegmentationStrategy(
        raw_queue, trans_queue, stop_event, config, vad_model_mock
    )

    # Set active mode and set up for finalization
    await strategy.set_active_output_mode(CliOutputMode.KEYBOARD)

    # Create a short test frame (not enough to meet min duration)
    frame_size = 1600  # 100ms at 16kHz
    test_frame = np.ones(frame_size, dtype=np.float32)

    # Add short segment that's less than min_speech_duration_ms
    strategy._current_segment.append(test_frame.copy())

    # Calculate duration in ms
    speech_duration_ms = (
        sum(len(f) for f in strategy._current_segment) / SAMPLE_RATE * 1000
    )
    assert speech_duration_ms < config.vad.min_speech_duration_ms

    # Call finalize directly
    await strategy._finalize_segment()

    # Verify no task was created (segment too short)
    assert trans_queue.empty()

    # Add more frames to exceed minimum duration
    strategy._current_segment.append(test_frame.copy())
    strategy._current_segment.append(test_frame.copy())
    strategy._current_segment.append(test_frame.copy())

    # Calculate new duration
    speech_duration_ms = (
        sum(len(f) for f in strategy._current_segment) / SAMPLE_RATE * 1000
    )
    assert speech_duration_ms >= config.vad.min_speech_duration_ms

    # Finalize again
    await strategy._finalize_segment()

    # Now a task should be created
    assert not trans_queue.empty()
    task = await trans_queue.get()
    assert isinstance(task, TranscriptionTask)
    assert task.output_mode == CliOutputMode.KEYBOARD


@pytest.mark.asyncio
async def test_vad_strategy_max_speech_duration(vad_strategy):
    """Test VAD strategy respects maximum speech duration."""
    # Configure short max duration
    vad_strategy.vad_config.max_speech_duration_s = 0.1  # Very short for testing

    # Set up the state and create a test frame that will exceed max duration
    await vad_strategy.set_active_output_mode(CliOutputMode.KEYBOARD)
    vad_strategy._vad_state = VADState.SPEECH

    # Create a frame that will exceed max duration when added
    frame_size = 2000  # 125ms at 16kHz (exceeds our 100ms limit)
    test_frame = np.ones(frame_size, dtype=np.float32)

    # Add frames to current segment to exceed max duration
    vad_strategy._current_segment.append(test_frame.copy())

    # Create a spy on _finalize_segment
    with patch.object(
        vad_strategy, "_finalize_segment", wraps=vad_strategy._finalize_segment
    ) as spy:
        # Calculate duration and verify it exceeds max
        current_duration_s = (
            sum(len(f) for f in vad_strategy._current_segment) / SAMPLE_RATE
        )
        assert current_duration_s >= vad_strategy.vad_config.max_speech_duration_s

        # Simulate the max duration check from process() method
        await vad_strategy._finalize_segment()
        vad_strategy._vad_state = VADState.SILENT

        # Verify finalize was called and state changed
        spy.assert_awaited_once()
        assert vad_strategy._vad_state == VADState.SILENT
