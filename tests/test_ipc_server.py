"""Tests for IPC server."""

import asyncio
import json
import pytest
import pytest_asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock, ANY

from handsfreed.state import DaemonStateManager, DaemonStateEnum
from handsfreed.ipc_server import IPCServer
from handsfreed.ipc_models import CliOutputMode


@pytest.fixture
def state_manager():
    """Create a state manager instance."""
    return DaemonStateManager()


@pytest.fixture
def socket_path(tmp_path):
    """Create a temporary socket path."""
    return tmp_path / "test.sock"


@pytest.fixture
def shutdown_event():
    """Create a shutdown event."""
    return asyncio.Event()


@pytest.fixture
def audio_capture():
    """Create a mock audio capture instance."""
    capture = AsyncMock()
    capture.start = AsyncMock()
    capture.stop = AsyncMock()
    return capture


@pytest.fixture
def transcriber():
    """Create a mock transcriber instance."""
    trans = AsyncMock()
    trans.start = AsyncMock(return_value=True)
    trans.stop = AsyncMock()
    trans.set_current_output_mode = MagicMock()
    trans.output_queue = asyncio.Queue()  # Real queue for output handler
    return trans


@pytest.fixture
def output_handler():
    """Create a mock output handler instance."""
    handler = AsyncMock()
    handler.start = AsyncMock()
    handler.stop = AsyncMock()
    return handler


@pytest_asyncio.fixture
async def ipc_server(
    socket_path,
    state_manager,
    shutdown_event,
    transcriber,
    audio_capture,
    output_handler,
):
    """Create an IPC server instance."""
    server = IPCServer(
        socket_path,
        state_manager,
        shutdown_event,
        transcriber,
        audio_capture,
        output_handler,
    )
    try:
        yield server
    finally:
        # Cleanup
        if server._server:
            await server.stop()


async def send_command_get_response(socket_path: Path, command: dict) -> dict:
    """Helper to send command and get response."""
    reader, writer = await asyncio.open_unix_connection(str(socket_path))
    try:
        # Send command
        command_json = json.dumps(command) + "\n"
        writer.write(command_json.encode())
        await writer.drain()

        # Read response
        response = await asyncio.wait_for(reader.readline(), timeout=1.0)
        return json.loads(response.decode())
    finally:
        writer.close()
        await writer.wait_closed()


@pytest.mark.asyncio
async def test_server_start(ipc_server, socket_path):
    """Test starting the server."""
    await ipc_server.start()
    assert ipc_server._server is not None
    assert socket_path.exists()
    assert socket_path.is_socket()


@pytest.mark.asyncio
async def test_server_stop(ipc_server, socket_path):
    """Test stopping the server."""
    await ipc_server.start()
    assert ipc_server._server is not None

    await ipc_server.stop()
    assert ipc_server._server is None
    assert not socket_path.exists()


@pytest.mark.asyncio
async def test_server_existing_socket(ipc_server, socket_path):
    """Test handling of existing socket file."""
    socket_path.touch()

    with pytest.raises(OSError, match="not a socket"):
        await ipc_server.start()


@pytest.mark.asyncio
async def test_start_command_success(
    ipc_server, socket_path, state_manager, transcriber, audio_capture, output_handler
):
    """Test successful Start command."""
    await ipc_server.start()

    # Send Start command
    response = await send_command_get_response(
        socket_path, {"command": "start", "output_mode": "keyboard"}
    )

    # Check response
    assert response["response_type"] == "ack"

    # Verify handlers were called in correct order
    output_handler.start.assert_awaited_once_with(transcriber.output_queue)
    transcriber.set_current_output_mode.assert_called_with(CliOutputMode.KEYBOARD)
    transcriber.start.assert_awaited_once()
    audio_capture.start.assert_awaited_once()

    # Check state
    assert state_manager.current_state == DaemonStateEnum.LISTENING


@pytest.mark.asyncio
async def test_start_command_transcriber_failure(
    ipc_server, socket_path, state_manager, transcriber, audio_capture, output_handler
):
    """Test Start command when transcriber fails."""
    await ipc_server.start()
    transcriber.start.return_value = False  # Make transcriber fail

    # Send Start command
    response = await send_command_get_response(
        socket_path, {"command": "start", "output_mode": "keyboard"}
    )

    # Check response
    assert response["response_type"] == "error"
    assert "Failed to start" in response["message"]

    # Check cleanup order
    output_handler.stop.assert_awaited_once()
    audio_capture.start.assert_not_awaited()  # Should not start audio

    # Check state
    assert state_manager.current_state == DaemonStateEnum.ERROR


@pytest.mark.asyncio
async def test_start_command_audio_failure(
    ipc_server, socket_path, state_manager, transcriber, audio_capture, output_handler
):
    """Test Start command when audio capture fails."""
    await ipc_server.start()
    audio_capture.start.side_effect = RuntimeError("Test error")

    # Send Start command
    response = await send_command_get_response(
        socket_path, {"command": "start", "output_mode": "keyboard"}
    )

    # Check response
    assert response["response_type"] == "error"
    assert "Failed to start" in response["message"]

    # Check cleanup order
    audio_capture.start.assert_awaited_once()  # Should attempt to start
    transcriber.stop.assert_awaited_once()  # Should stop transcriber
    output_handler.stop.assert_awaited_once()  # Should stop output

    # Check state
    assert state_manager.current_state == DaemonStateEnum.ERROR


@pytest.mark.asyncio
async def test_start_while_running(
    ipc_server, socket_path, state_manager, transcriber, audio_capture, output_handler
):
    """Test Start command while already running."""
    await ipc_server.start()
    state_manager.set_state(DaemonStateEnum.LISTENING)

    # Send Start command with different mode
    response = await send_command_get_response(
        socket_path, {"command": "start", "output_mode": "clipboard"}
    )

    # Should succeed but only change mode
    assert response["response_type"] == "ack"
    transcriber.set_current_output_mode.assert_called_with(CliOutputMode.CLIPBOARD)
    output_handler.start.assert_not_awaited()  # Should not restart
    transcriber.start.assert_not_awaited()
    audio_capture.start.assert_not_awaited()


@pytest.mark.asyncio
async def test_stop_command_success(
    ipc_server, socket_path, state_manager, transcriber, audio_capture, output_handler
):
    """Test successful Stop command."""
    await ipc_server.start()
    state_manager.set_state(DaemonStateEnum.LISTENING)

    # Send Stop command
    response = await send_command_get_response(socket_path, {"command": "stop"})

    # Check response
    assert response["response_type"] == "ack"

    # Verify handlers were stopped in correct order
    audio_capture.stop.assert_awaited_once()
    transcriber.stop.assert_awaited_once()
    output_handler.stop.assert_awaited_once()

    # Check state
    assert state_manager.current_state == DaemonStateEnum.IDLE


@pytest.mark.asyncio
async def test_stop_when_idle(
    ipc_server, socket_path, transcriber, audio_capture, output_handler
):
    """Test Stop command when already idle."""
    await ipc_server.start()

    # Send Stop command
    response = await send_command_get_response(socket_path, {"command": "stop"})

    # Should succeed but do nothing
    assert response["response_type"] == "ack"
    output_handler.stop.assert_not_awaited()
    transcriber.stop.assert_not_awaited()
    audio_capture.stop.assert_not_awaited()


@pytest.mark.asyncio
async def test_status_command(ipc_server, socket_path, state_manager):
    """Test Status command."""
    await ipc_server.start()
    state_manager.set_state(DaemonStateEnum.LISTENING)

    # Send Status command
    response = await send_command_get_response(socket_path, {"command": "status"})

    # Check response
    assert response["response_type"] == "status"
    assert response["status"]["state"] == "listening"
    assert response["status"]["last_error"] is None


@pytest.mark.asyncio
async def test_shutdown_command(ipc_server, socket_path, shutdown_event):
    """Test Shutdown command."""
    await ipc_server.start()

    # Send Shutdown command
    response = await send_command_get_response(socket_path, {"command": "shutdown"})

    # Check response and shutdown signal
    assert response["response_type"] == "ack"
    assert shutdown_event.is_set()


@pytest.mark.asyncio
async def test_invalid_command(ipc_server, socket_path):
    """Test handling of invalid command."""
    await ipc_server.start()

    # Send invalid command
    response = await send_command_get_response(socket_path, {"command": "invalid"})

    assert response["response_type"] == "error"
    assert "Invalid command format" in response["message"]
