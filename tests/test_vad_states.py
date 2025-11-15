import asyncio
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest
from handsfreed.segmentation.vad import (
    SilentState,
    SpeechState,
    EndingSpeechState,
    AbstractVADState,
)


@pytest.fixture
def strategy_mock():
    """Create a mock VADSegmentationStrategy."""
    strategy = AsyncMock()
    strategy.vad_config.threshold = 0.5
    strategy.vad_config.neg_threshold = 0.3
    strategy.vad_config.max_speech_duration_s = 10.0
    strategy.vad_config.min_silence_duration_ms = 100
    strategy._pre_roll_buffer = [np.ones(512)]
    strategy._current_segment = []
    return strategy


@pytest.mark.asyncio
async def test_silent_state_speech_detected(strategy_mock):
    """Test SilentState transitions to SpeechState when speech is detected."""
    state = SilentState()
    new_state = await state.handle_vad_result(strategy_mock, 0.7, np.ones(512))
    assert isinstance(new_state, SpeechState)
    assert len(strategy_mock._current_segment) == 1


@pytest.mark.asyncio
async def test_silent_state_no_speech(strategy_mock):
    """Test SilentState remains in SilentState when no speech is detected."""
    state = SilentState()
    new_state = await state.handle_vad_result(strategy_mock, 0.2, np.ones(512))
    assert isinstance(new_state, SilentState)
    assert len(strategy_mock._current_segment) == 0


@pytest.mark.asyncio
async def test_speech_state_no_speech(strategy_mock):
    """Test SpeechState transitions to EndingSpeechState when silence is detected."""
    state = SpeechState()
    new_state = await state.handle_vad_result(strategy_mock, 0.2, np.ones(512))
    assert isinstance(new_state, EndingSpeechState)
    assert len(strategy_mock._current_segment) == 1


@pytest.mark.asyncio
async def test_speech_state_speech_continues(strategy_mock):
    """Test SpeechState remains in SpeechState when speech continues."""
    state = SpeechState()
    new_state = await state.handle_vad_result(strategy_mock, 0.7, np.ones(512))
    assert isinstance(new_state, SpeechState)
    assert len(strategy_mock._current_segment) == 1


@pytest.mark.asyncio
async def test_speech_state_max_duration(strategy_mock):
    """Test SpeechState transitions to SilentState when max speech duration is reached."""
    strategy_mock.vad_config.max_speech_duration_s = 0.1
    state = SpeechState()
    # Add enough frames to exceed max duration
    for _ in range(10):
        strategy_mock._current_segment.append(np.ones(2000))
    new_state = await state.handle_vad_result(strategy_mock, 0.7, np.ones(512))
    assert isinstance(new_state, SilentState)
    strategy_mock._finalize_segment.assert_awaited_once()


@pytest.mark.asyncio
async def test_ending_speech_state_speech_resumes(strategy_mock):
    """Test EndingSpeechState transitions back to SpeechState when speech resumes."""
    state = EndingSpeechState()
    new_state = await state.handle_vad_result(strategy_mock, 0.7, np.ones(512))
    assert isinstance(new_state, SpeechState)
    assert len(strategy_mock._current_segment) == 1


@pytest.mark.asyncio
async def test_ending_speech_state_silence_continues(strategy_mock):
    """Test EndingSpeechState remains in EndingSpeechState when silence continues."""
    state = EndingSpeechState()
    new_state = await state.handle_vad_result(strategy_mock, 0.2, np.ones(512))
    assert isinstance(new_state, EndingSpeechState)
    assert len(strategy_mock._current_segment) == 1


@pytest.mark.asyncio
async def test_ending_speech_state_silence_timeout(strategy_mock):
    """Test EndingSpeechState transitions to SilentState after silence timeout."""
    state = EndingSpeechState()
    await asyncio.sleep(0.2)  # Wait for silence timeout
    new_state = await state.handle_vad_result(strategy_mock, 0.2, np.ones(512))
    assert isinstance(new_state, SilentState)
    strategy_mock._finalize_segment.assert_awaited_once()
