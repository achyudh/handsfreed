"""Tests for IPC command and response models."""

import json
import pytest
from pydantic import ValidationError

from handsfreed.ipc_models import (
    CliOutputMode,
    StartCommand,
    StopCommand,
    StatusCommand,
    ShutdownCommand,
    CommandWrapper,
    DaemonStateModel,
    AckResponse,
    StatusResponse,
    ErrorResponse,
    ResponseWrapper,
)


def test_start_command_validation():
    """Test StartCommand validation."""
    # Valid command
    cmd = StartCommand(output_mode=CliOutputMode.KEYBOARD)
    assert cmd.command == "start"
    assert cmd.output_mode == CliOutputMode.KEYBOARD

    # Invalid output mode
    with pytest.raises(ValidationError):
        StartCommand(output_mode="invalid")


def test_command_parsing():
    """Test parsing different commands from JSON."""
    # Start command
    start_json = '{"command": "start", "output_mode": "keyboard"}'
    cmd = CommandWrapper.model_validate_json(start_json)
    assert isinstance(cmd.root, StartCommand)
    assert cmd.output_mode == CliOutputMode.KEYBOARD

    # Stop command
    stop_json = '{"command": "stop"}'
    cmd = CommandWrapper.model_validate_json(stop_json)
    assert isinstance(cmd.root, StopCommand)

    # Status command
    status_json = '{"command": "status"}'
    cmd = CommandWrapper.model_validate_json(status_json)
    assert isinstance(cmd.root, StatusCommand)

    # Shutdown command
    shutdown_json = '{"command": "shutdown"}'
    cmd = CommandWrapper.model_validate_json(shutdown_json)
    assert isinstance(cmd.root, ShutdownCommand)


def test_invalid_command_parsing():
    """Test handling of invalid commands."""
    # Invalid command type
    with pytest.raises(ValidationError):
        CommandWrapper.model_validate_json('{"command": "invalid"}')

    # Missing required field
    with pytest.raises(ValidationError):
        CommandWrapper.model_validate_json('{"command": "start"}')

    # Invalid JSON syntax
    with pytest.raises(ValidationError, match="Invalid JSON"):
        CommandWrapper.model_validate_json("invalid json")


def test_response_serialization():
    """Test serializing different responses to JSON."""
    # Ack response
    ack = ResponseWrapper(root=AckResponse())
    ack_json = ack.model_dump_json()
    assert json.loads(ack_json) == {"response_type": "ack"}

    # Status response
    status = ResponseWrapper(
        root=StatusResponse(status=DaemonStateModel(state="idle", last_error=None))
    )
    status_json = status.model_dump_json()
    assert json.loads(status_json) == {
        "response_type": "status",
        "status": {"state": "idle", "last_error": None},
    }

    # Error response
    error = ResponseWrapper(root=ErrorResponse(message="Test error"))
    error_json = error.model_dump_json()
    assert json.loads(error_json) == {"response_type": "error", "message": "Test error"}


def test_response_parsing():
    """Test parsing different responses from JSON."""
    # Ack response
    ack_json = '{"response_type": "ack"}'
    resp = ResponseWrapper.model_validate_json(ack_json)
    assert isinstance(resp.root, AckResponse)

    # Status response
    status_json = '{"response_type": "status", "status": {"state": "idle"}}'
    resp = ResponseWrapper.model_validate_json(status_json)
    assert isinstance(resp.root, StatusResponse)
    assert resp.root.status.state == "idle"

    # Error response
    error_json = '{"response_type": "error", "message": "Test error"}'
    resp = ResponseWrapper.model_validate_json(error_json)
    assert isinstance(resp.root, ErrorResponse)
    assert resp.root.message == "Test error"


def test_invalid_response_parsing():
    """Test handling of invalid responses."""
    # Invalid response type
    with pytest.raises(ValueError, match="Invalid response_type"):
        ResponseWrapper.model_validate_json('{"response_type": "invalid"}')

    # Missing required field
    with pytest.raises(ValidationError):
        ResponseWrapper.model_validate_json('{"response_type": "error"}')


def test_wrapper_attribute_delegation():
    """Test attribute delegation in wrapper models."""
    # Command wrapper
    cmd = CommandWrapper(root=StartCommand(output_mode=CliOutputMode.KEYBOARD))
    assert cmd.command == "start"
    assert cmd.output_mode == CliOutputMode.KEYBOARD

    # Response wrapper
    resp = ResponseWrapper(root=ErrorResponse(message="test"))
    assert resp.response_type == "error"
    assert resp.message == "test"
