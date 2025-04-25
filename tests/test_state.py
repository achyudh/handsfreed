"""Tests for the state management module."""

import pytest

from handsfreed.state import DaemonStateEnum, DaemonStateManager


def test_initial_state():
    """Test initial state is IDLE with no error."""
    manager = DaemonStateManager()
    assert manager.current_state == DaemonStateEnum.IDLE
    assert manager.last_error is None

    status_state, status_error = manager.get_status()
    assert status_state == "Idle"
    assert status_error is None


def test_set_state():
    """Test setting different states."""
    manager = DaemonStateManager()

    manager.set_state(DaemonStateEnum.LISTENING)
    assert manager.current_state == DaemonStateEnum.LISTENING
    assert manager.last_error is None

    manager.set_state(DaemonStateEnum.PROCESSING)
    assert manager.current_state == DaemonStateEnum.PROCESSING
    assert manager.last_error is None

    status_state, status_error = manager.get_status()
    assert status_state == "Processing"
    assert status_error is None


def test_set_invalid_state():
    """Test error handling when setting invalid state."""
    manager = DaemonStateManager()
    with pytest.raises(TypeError, match="must be a DaemonStateEnum"):
        manager.set_state("LISTENING")  # type: ignore


def test_set_error():
    """Test setting error state and message."""
    manager = DaemonStateManager()
    error_message = "Test error message"
    manager.set_error(error_message)

    assert manager.current_state == DaemonStateEnum.ERROR
    assert manager.last_error == error_message

    status_state, status_error = manager.get_status()
    assert status_state == "Error"
    assert status_error == error_message


def test_error_cleared_on_state_change():
    """Test that error message is cleared when moving out of error state."""
    manager = DaemonStateManager()

    # Set error first
    manager.set_error("Test error")
    assert manager.last_error == "Test error"

    # Change to another state
    manager.set_state(DaemonStateEnum.IDLE)
    assert manager.current_state == DaemonStateEnum.IDLE
    assert manager.last_error is None


def test_state_transitions():
    """Test various state transitions."""
    manager = DaemonStateManager()
    transitions = [
        (DaemonStateEnum.LISTENING, None),
        (DaemonStateEnum.PROCESSING, None),
        (DaemonStateEnum.ERROR, "Error occurred"),
        (DaemonStateEnum.IDLE, None),
    ]

    for new_state, error in transitions:
        if error:
            manager.set_error(error)
        else:
            manager.set_state(new_state)

        state, err = manager.get_status()
        assert state == new_state.value
        assert err == error
