"""Tests for IPC server."""

import asyncio
import json
import pytest
import pytest_asyncio
from pathlib import Path

from handsfreed.state import DaemonStateManager, DaemonStateEnum
from handsfreed.ipc_server import IPCServer


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


@pytest_asyncio.fixture
async def ipc_server(socket_path, state_manager, shutdown_event):
    """Create an IPC server instance."""
    server = IPCServer(socket_path, state_manager, shutdown_event)
    try:
        yield server
    finally:
        # Cleanup
        if server._server:
            await server.stop()


async def send_command_get_response(socket_path: Path, command_dict: dict) -> dict:
    """Helper to send a command and get response."""
    reader, writer = await asyncio.open_unix_connection(str(socket_path))
    try:
        # Send command
        command_json = json.dumps(command_dict) + "\n"
        writer.write(command_json.encode())
        await writer.drain()

        # Read response
        response = await reader.readline()
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
async def test_status_command(ipc_server: IPCServer, socket_path, state_manager):
    """Test status command returns current state."""
    await ipc_server.start()

    # Set a known state
    state_manager.set_state(DaemonStateEnum.LISTENING)

    # Send status command
    response = await send_command_get_response(socket_path, {"command": "status"})

    assert response["response_type"] == "status"
    assert response["status"]["state"] == "listening"
    assert response["status"]["last_error"] is None


@pytest.mark.asyncio
async def test_status_command_with_error(ipc_server, socket_path, state_manager):
    """Test status command includes error message."""
    await ipc_server.start()

    # Set error state
    error_msg = "Test error"
    state_manager.set_error(error_msg)

    # Send status command
    response = await send_command_get_response(socket_path, {"command": "status"})

    assert response["response_type"] == "status"
    assert response["status"]["state"] == "error"
    assert response["status"]["last_error"] == error_msg


@pytest.mark.asyncio
async def test_shutdown_command(ipc_server, socket_path, shutdown_event):
    """Test shutdown command sets event."""
    await ipc_server.start()

    # Send shutdown command
    response = await send_command_get_response(socket_path, {"command": "shutdown"})

    assert response["response_type"] == "ack"
    assert shutdown_event.is_set()


@pytest.mark.asyncio
async def test_invalid_command_json(ipc_server, socket_path):
    """Test handling of invalid JSON."""
    await ipc_server.start()

    reader, writer = await asyncio.open_unix_connection(str(socket_path))
    try:
        # Send invalid JSON
        writer.write(b"invalid json\n")
        await writer.drain()

        # Read response
        response = await reader.readline()
        response_dict = json.loads(response.decode())

        assert response_dict["response_type"] == "error"
        assert "Invalid JSON" in response_dict["message"]
    finally:
        writer.close()
        await writer.wait_closed()


@pytest.mark.asyncio
async def test_invalid_command_format(ipc_server, socket_path):
    """Test handling of invalid command format."""
    await ipc_server.start()

    # Send command with invalid structure
    response = await send_command_get_response(socket_path, {"command": "invalid"})

    assert response["response_type"] == "error"
    assert "Invalid command format" in response["message"]


@pytest.mark.asyncio
async def test_connection_cleanup(ipc_server, socket_path):
    """Test server cleans up client tasks."""
    await ipc_server.start()
    initial_tasks = len(ipc_server._client_tasks)

    # Open and close a connection
    reader, writer = await asyncio.open_unix_connection(str(socket_path))
    writer.close()
    await writer.wait_closed()

    # Wait briefly for cleanup
    await asyncio.sleep(0.1)
    assert len(ipc_server._client_tasks) == initial_tasks
