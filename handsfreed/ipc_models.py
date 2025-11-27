"""IPC command and response models for handsfreed daemon."""

from enum import Enum
from typing import Literal, Optional, Union, Annotated
from pydantic import BaseModel, Field, RootModel


class CliOutputMode(str, Enum):
    """Output mode for transcribed text."""

    KEYBOARD = "keyboard"
    CLIPBOARD = "clipboard"


class StartCommand(BaseModel):
    """Command to start transcription."""

    command: Literal["start"] = "start"
    output_mode: CliOutputMode


class StopCommand(BaseModel):
    """Command to stop transcription."""

    command: Literal["stop"] = "stop"


class StatusCommand(BaseModel):
    """Command to get daemon status."""

    command: Literal["status"] = "status"


class ShutdownCommand(BaseModel):
    """Command to shut down the daemon."""

    command: Literal["shutdown"] = "shutdown"


class ToggleCommand(BaseModel):
    """Command to toggle transcription state."""

    command: Literal["toggle"] = "toggle"
    # Optional output mode. If None, uses default/current.
    output_mode: Optional[CliOutputMode] = None


class SubscribeCommand(BaseModel):
    """Command to subscribe to state change events."""

    command: Literal["subscribe"] = "subscribe"


# Use discriminated union for command parsing
DaemonCommand = Annotated[
    Union[
        StartCommand,
        StopCommand,
        StatusCommand,
        ShutdownCommand,
        ToggleCommand,
        SubscribeCommand,
    ],
    Field(discriminator="command"),
]


# Wrapper for easy command parsing using RootModel
class CommandWrapper(RootModel[DaemonCommand]):
    """Wrapper model for parsing incoming commands."""

    root: DaemonCommand

    def __getattr__(self, name: str):
        """Delegate attribute access to the root command."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.root, name)


class DaemonStateModel(BaseModel):
    """Model representing daemon state."""

    state: str
    last_error: Optional[str] = None


class AckResponse(BaseModel):
    """Simple acknowledgment response."""

    response_type: Literal["ack"] = "ack"


class StatusResponse(BaseModel):
    """Response containing daemon status."""

    response_type: Literal["status"] = "status"
    status: DaemonStateModel


class ErrorResponse(BaseModel):
    """Response indicating an error."""

    response_type: Literal["error"] = "error"
    message: str


class StateNotification(BaseModel):
    """Notification broadcast when daemon state changes."""

    response_type: Literal["state_change"] = "state_change"
    status: DaemonStateModel


# Use discriminated union for response serialization
DaemonResponse = Annotated[
    Union[AckResponse, StatusResponse, ErrorResponse, StateNotification],
    Field(discriminator="response_type"),
]


# Wrapper for easy response serialization using RootModel
class ResponseWrapper(RootModel[DaemonResponse]):
    """Wrapper model for serializing outgoing responses."""

    root: DaemonResponse

    def __getattr__(self, name: str):
        """Delegate attribute access to the root response."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.root, name)

    def model_dump_json(self, **kwargs) -> str:
        """Override to unwrap the response for serialization."""
        return self.root.model_dump_json(**kwargs)

    @classmethod
    def model_validate_json(cls, json_data: str, **kwargs):
        """Override to wrap the parsed response data."""
        # Parse as raw JSON first
        import json

        try:
            data = json.loads(json_data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}")

        # Find the response type and validate
        response_type = data.get("response_type")
        if not response_type:
            raise ValueError("Missing response_type field")

        # Create the appropriate response object
        response_types = {
            "ack": AckResponse,
            "status": StatusResponse,
            "error": ErrorResponse,
            "state_change": StateNotification,
        }

        if response_type not in response_types:
            raise ValueError(f"Invalid response_type: {response_type}")

        # Parse into the specific response type first
        response_model = response_types[response_type]
        response = response_model.model_validate(data)

        # Wrap in ResponseWrapper
        return cls(root=response)
