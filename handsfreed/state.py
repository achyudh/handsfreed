"""State management for the handsfreed daemon."""

from enum import Enum
from typing import Any, Callable, List, Optional, Tuple


class DaemonStateEnum(str, Enum):
    """Possible states for the daemon."""

    IDLE = "Idle"
    LISTENING = "Listening"
    PROCESSING = "Processing"
    ERROR = "Error"


class DaemonStateManager:
    """Manages the operational state of the daemon."""

    def __init__(self):
        """Initialize state manager with IDLE state."""
        self._state: DaemonStateEnum = DaemonStateEnum.IDLE
        self._last_error: Optional[str] = None
        self._observers: List[Callable[[DaemonStateEnum, Optional[str]], Any]] = []

    @property
    def current_state(self) -> DaemonStateEnum:
        """Get the current state of the daemon."""
        return self._state

    @property
    def last_error(self) -> Optional[str]:
        """Get the last error message, if any."""
        return self._last_error

    def add_observer(
        self, observer: Callable[[DaemonStateEnum, Optional[str]], Any]
    ) -> None:
        """Add an observer callback for state changes.

        The callback receives the new state and optional error message.
        """
        self._observers.append(observer)

    def _notify_observers(self) -> None:
        """Notify all observers of the current state."""
        for observer in self._observers:
            try:
                observer(self._state, self._last_error)
            except Exception:
                # Don't let observer errors break state management
                pass

    def set_state(self, new_state: DaemonStateEnum) -> None:
        """Set the daemon's operational state.

        Args:
            new_state: The new state to set.

        Raises:
            TypeError: If the provided state is not a valid DaemonStateEnum.
        """
        if not isinstance(new_state, DaemonStateEnum):
            raise TypeError(f"State must be a DaemonStateEnum, got {type(new_state)}")

        # Reset error when moving out of error state
        if new_state != DaemonStateEnum.ERROR:
            self._last_error = None

        if self._state != new_state:
            self._state = new_state
            self._notify_observers()

    def set_error(self, message: str) -> None:
        """Set an error state with the provided message.

        Args:
            message: The error message to store.
        """
        changed = self._state != DaemonStateEnum.ERROR or self._last_error != message
        self._last_error = message
        self._state = DaemonStateEnum.ERROR

        if changed:
            self._notify_observers()

    def get_status(self) -> Tuple[str, Optional[str]]:
        """Get the current status and last error message.

        Returns:
            A tuple containing the current state value (as a string) and the last error
            message (if any).</parameter_text">
        """
        return self.current_state.value, self.last_error
