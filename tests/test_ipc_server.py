"""Tests for IPC server."""

import asyncio
import pytest
import pytest_asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from handsfreed.state import DaemonStateManager
from handsfreed.ipc_server import IPCServer


@pytest.fixture
def state_manager():
    """Create a state manager instance."""
    return DaemonStateManager()


@pytest.fixture
def socket_path(tmp_path):
    """Create a temporary socket path."""
    return tmp_path / "test.sock"


@pytest_asyncio.fixture
async def ipc_server(socket_path, state_manager):
    """Create an IPC server instance."""
    server = IPCServer(socket_path, state_manager)
    try:
        yield server
    finally:
        # Cleanup
        if server._server:
            await server.stop()


@pytest.mark.asyncio
async def test_server_start(ipc_server: IPCServer, socket_path):
    """Test starting the server."""
    await ipc_server.start()
    assert ipc_server._server is not None
    assert socket_path.exists()
    assert socket_path.is_socket()


@pytest.mark.asyncio
async def test_server_stop(ipc_server: IPCServer, socket_path):
    """Test stopping the server."""
    await ipc_server.start()
    assert ipc_server._server is not None

    await ipc_server.stop()
    assert ipc_server._server is None
    assert not socket_path.exists()


@pytest.mark.asyncio
async def test_server_existing_socket(ipc_server: IPCServer, socket_path):
    """Test handling of existing socket file."""
    # Create a file at the socket path
    socket_path.touch()

    # Should fail if path exists but isn't a socket
    with pytest.raises(OSError, match="not a socket"):
        await ipc_server.start()


@pytest.mark.asyncio
async def test_server_client_echo(ipc_server: IPCServer, socket_path):
    """Test basic client communication (echo)."""
    await ipc_server.start()

    # Connect a test client
    reader, writer = await asyncio.open_unix_connection(str(socket_path))

    try:
        # Send test message
        test_msg = "Hello, server!\n"
        writer.write(test_msg.encode())
        await writer.drain()

        # Read response
        response = await asyncio.wait_for(reader.readline(), timeout=1.0)
        assert response.decode().strip() == f"Echo: {test_msg.strip()}"

    finally:
        writer.close()
        await writer.wait_closed()


@pytest.mark.asyncio
async def test_server_multiple_clients(ipc_server: IPCServer, socket_path):
    """Test handling multiple client connections."""
    await ipc_server.start()

    # Connect two test clients
    reader1, writer1 = await asyncio.open_unix_connection(str(socket_path))
    reader2, writer2 = await asyncio.open_unix_connection(str(socket_path))

    try:
        # Send messages from both clients
        msg1 = "Client 1\n"
        msg2 = "Client 2\n"

        writer1.write(msg1.encode())
        writer2.write(msg2.encode())
        await writer1.drain()
        await writer2.drain()

        # Read responses
        response1 = await asyncio.wait_for(reader1.readline(), timeout=1.0)
        response2 = await asyncio.wait_for(reader2.readline(), timeout=1.0)

        assert response1.decode().strip() == f"Echo: {msg1.strip()}"
        assert response2.decode().strip() == f"Echo: {msg2.strip()}"

    finally:
        writer1.close()
        writer2.close()
        await writer1.wait_closed()
        await writer2.wait_closed()


@pytest.mark.asyncio
async def test_server_client_disconnect(ipc_server: IPCServer, socket_path):
    """Test handling client disconnection."""
    await ipc_server.start()
    initial_tasks = len(ipc_server._client_tasks)

    # Connect and immediately disconnect
    reader, writer = await asyncio.open_unix_connection(str(socket_path))
    writer.close()
    await writer.wait_closed()

    # Give the server time to clean up
    await asyncio.sleep(0.1)

    # Should be back to initial task count
    assert len(ipc_server._client_tasks) == initial_tasks
