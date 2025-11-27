"""Tests for the TaskAssembler component."""

import asyncio
import numpy as np
import pytest
import pytest_asyncio
from unittest.mock import Mock

from handsfreed.task_assembler import TaskAssembler
from handsfreed.ipc_models import CliOutputMode
from handsfreed.pipeline import TranscriptionTask


@pytest_asyncio.fixture
async def segment_queue():
    """Create a queue for input segments."""
    queue = asyncio.Queue()
    yield queue
    # Drain queue
    while not queue.empty():
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            break


@pytest_asyncio.fixture
async def transcription_queue():
    """Create a queue for output tasks."""
    queue = asyncio.Queue()
    yield queue
    # Drain queue
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
async def task_assembler(segment_queue, transcription_queue, stop_event):
    """Create a TaskAssembler instance."""
    assembler = TaskAssembler(segment_queue, transcription_queue, stop_event)
    yield assembler
    await assembler.stop()


@pytest.mark.asyncio
async def test_init(task_assembler, segment_queue, transcription_queue):
    """Test initialization."""
    assert task_assembler.input_queue is segment_queue
    assert task_assembler.output_queue is transcription_queue
    assert task_assembler._active_mode is None


@pytest.mark.asyncio
async def test_set_output_mode(task_assembler):
    """Test setting the output mode."""
    # Initial state
    assert task_assembler._active_mode is None

    # Set to KEYBOARD
    task_assembler.set_output_mode(CliOutputMode.KEYBOARD)
    assert task_assembler._active_mode == CliOutputMode.KEYBOARD

    # Set to CLIPBOARD
    task_assembler.set_output_mode(CliOutputMode.CLIPBOARD)
    assert task_assembler._active_mode == CliOutputMode.CLIPBOARD

    # Set to None
    task_assembler.set_output_mode(None)
    assert task_assembler._active_mode is None


@pytest.mark.asyncio
async def test_process_segment_with_mode(
    task_assembler, segment_queue, transcription_queue
):
    """Test processing a segment when a mode is set."""
    # Set mode
    task_assembler.set_output_mode(CliOutputMode.KEYBOARD)
    await task_assembler.start()

    # Create a dummy audio segment
    audio_segment = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    # Put into input queue
    await segment_queue.put(audio_segment)

    # Wait for processing
    task = await asyncio.wait_for(transcription_queue.get(), timeout=1.0)

    # Verify task
    assert isinstance(task, TranscriptionTask)
    assert np.array_equal(task.audio, audio_segment)
    assert task.output_mode == CliOutputMode.KEYBOARD


@pytest.mark.asyncio
async def test_process_segment_without_mode(
    task_assembler, segment_queue, transcription_queue
):
    """Test dropping a segment when no mode is set."""
    # Ensure mode is None
    task_assembler.set_output_mode(None)
    await task_assembler.start()

    # Create a dummy audio segment
    audio_segment = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    # Put into input queue
    await segment_queue.put(audio_segment)

    # Wait briefly to ensure nothing is produced
    await asyncio.sleep(0.1)

    assert transcription_queue.empty()


@pytest.mark.asyncio
async def test_process_multiple_segments_changing_modes(
    task_assembler, segment_queue, transcription_queue
):
    """Test processing multiple segments while changing modes."""
    await task_assembler.start()

    segment1 = np.array([1.0], dtype=np.float32)
    segment2 = np.array([2.0], dtype=np.float32)
    segment3 = np.array([3.0], dtype=np.float32)

    # 1. Keyboard Mode
    task_assembler.set_output_mode(CliOutputMode.KEYBOARD)
    await segment_queue.put(segment1)

    task1 = await asyncio.wait_for(transcription_queue.get(), timeout=1.0)
    assert task1.output_mode == CliOutputMode.KEYBOARD
    assert np.array_equal(task1.audio, segment1)

    # 2. No Mode (should drop)
    task_assembler.set_output_mode(None)
    await segment_queue.put(segment2)
    await asyncio.sleep(0.1)
    assert transcription_queue.empty()

    # 3. Clipboard Mode
    task_assembler.set_output_mode(CliOutputMode.CLIPBOARD)
    await segment_queue.put(segment3)

    task3 = await asyncio.wait_for(transcription_queue.get(), timeout=1.0)
    assert task3.output_mode == CliOutputMode.CLIPBOARD
    assert np.array_equal(task3.audio, segment3)
