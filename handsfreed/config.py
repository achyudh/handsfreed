"""Configuration handling for handsfreed daemon."""

import getpass
import os
import tomllib
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, field_validator


def get_default_config_path() -> Path:
    """Get the default config file path following XDG spec."""
    xdg_config = os.environ.get("XDG_CONFIG_HOME")
    if xdg_config:
        base_dir = Path(xdg_config)
    else:
        base_dir = Path.home() / ".config"

    return base_dir / "handsfree" / "config.toml"


def get_default_socket_path() -> Path:
    """Get the default socket path following XDG spec."""
    xdg_runtime_dir = os.environ.get("XDG_RUNTIME_DIR")
    if xdg_runtime_dir:
        sock_dir = Path(xdg_runtime_dir) / "handsfree"
        try:
            sock_dir.mkdir(parents=True, exist_ok=True)
            if not os.access(sock_dir, os.W_OK | os.X_OK):
                raise OSError("Insufficient permissions for XDG runtime dir.")
            return sock_dir / "daemon.sock"
        except (OSError, PermissionError) as e:
            print(
                f"Warning: Could not use XDG_RUNTIME_DIR ({e}), falling back to /tmp."
            )
            pass

    # Fallback if XDG_RUNTIME_DIR not set or unusable
    uid = getpass.getuser()
    return Path(f"/tmp/handsfree-{uid}.sock")


def get_default_log_path() -> Path:
    """Get the default log file path following XDG spec."""
    xdg_state = os.environ.get("XDG_STATE_HOME")
    if xdg_state:
        base_dir = Path(xdg_state)
    else:
        base_dir = Path.home() / ".local" / "state"

    log_dir = base_dir / "handsfree"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / "handsfreed.log"


class AudioConfig(BaseModel):
    """Audio processing configuration."""

    input_gain: float = Field(
        default=1.0, gt=0, description="Input gain multiplier (1.0 = no gain)."
    )
    dc_offset_correction: bool = Field(
        default=True, description="Enable DC offset correction for raw audio."
    )
    dc_offset_window_ms: int = Field(
        default=512, ge=0, description="Window size for DC offset calculation (ms)."
    )


class VadConfig(BaseModel):
    """Voice Activity Detection configuration."""

    enabled: bool = Field(default=False, description="Enable Voice Activity Detection.")
    threshold: float = Field(
        default=0.5, description="Speech probability threshold (0.0-1.0)."
    )
    min_speech_duration_ms: int = Field(
        default=256, ge=1, description="Minimum duration for a speech segment (ms)."
    )
    min_silence_duration_ms: int = Field(
        default=1024,
        ge=1,
        description="Minimum duration of silence to end a speech segment (ms).",
    )
    pre_roll_duration_ms: int = Field(
        default=192,
        ge=0,
        description="Pre-roll duration to include before a detected speech segment (ms).",
    )
    neg_threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Negative threshold for speech detection (0.0-1.0, optional).",
    )
    max_speech_duration_s: float = Field(
        default=0.0,
        ge=0,
        description="Maximum duration of a single speech segment in seconds (0 = unlimited).",
    )
    auto_disable_duration_s: float = Field(
        default=5.0,
        ge=0,
        description="Maximum duration in seconds before listening stops (0 = disabled).",
    )


class WhisperConfig(BaseModel):
    """Whisper model configuration."""

    model: str = Field(
        default="small.en",
        description="Whisper model identifier (e.g., small.en, medium.en).",
    )
    device: str = Field(
        default="auto", description="Device for inference (auto, cpu, cuda)."
    )
    compute_type: str = Field(
        default="auto",
        description="Compute type for inference (auto, float32, float16, int8).",
    )
    language: Optional[str] = Field(
        default=None, description="Optional language code (empty for auto-detect)."
    )
    beam_size: int = Field(
        default=5,
        ge=1,
        description="Beam size for search (1-10, higher is slower but more accurate).",
    )
    cpu_threads: int = Field(
        default=0, ge=0, description="Number of CPU threads for inference (0 = auto)."
    )

    @field_validator("model")
    @classmethod
    def check_model_not_empty(cls, v: str) -> str:
        if not v:
            raise ValueError("Whisper model identifier cannot be empty")
        return v


class OutputConfig(BaseModel):
    """Output command configuration."""

    keyboard_command: Optional[str] = Field(
        default=None, description="Command to execute for keyboard output."
    )
    clipboard_command: Optional[str] = Field(
        default=None, description="Command to execute for clipboard output."
    )


class DaemonConfig(BaseModel):
    """Daemon runtime configuration."""

    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).",
    )
    log_file: Optional[Path] = Field(
        default=None, description="Optional custom log file path."
    )
    socket_path: Optional[Path] = Field(
        default=None, description="Optional custom socket path for IPC."
    )
    time_chunk_s: float = Field(
        default=5.0, gt=0, description="Duration of audio processing chunks in seconds."
    )

    @field_validator("log_level")
    @classmethod
    def check_log_level(cls, v: str) -> str:
        allowed_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper_v = v.upper()
        if upper_v not in allowed_levels:
            raise ValueError(f"Invalid log level. Choose from {allowed_levels}")
        return upper_v

    @property
    def computed_log_file(self) -> Path:
        return self.log_file or get_default_log_path()

    @property
    def computed_socket_path(self) -> Path:
        return self.socket_path or get_default_socket_path()


class AppConfig(BaseModel):
    """Root configuration."""

    audio: AudioConfig = Field(default_factory=AudioConfig)
    whisper: WhisperConfig = Field(default_factory=WhisperConfig)
    vad: VadConfig = Field(default_factory=VadConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    daemon: DaemonConfig = Field(default_factory=DaemonConfig)


def load_config(path: Optional[Path] = None) -> AppConfig:
    """Load and validate configuration.

    If path is not provided, looks for config in standard locations.
    If no config file is found, returns default configuration.

    Args:
        path: Optional path to config file.

    Returns:
        Validated AppConfig instance.

    Raises:
        ValueError: If config file exists but has invalid format/content.
        OSError: If config file exists but can't be read.
    """
    if path is None:
        path = get_default_config_path()

    if not path.exists():
        return AppConfig()  # Use defaults

    try:
        with open(path, "rb") as f:
            config_data = tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise ValueError(f"Error decoding TOML file: {path}\n{e}") from e
    except OSError as e:
        raise OSError(f"Error reading file: {path}\n{e}") from e

    try:
        return AppConfig(**config_data)
    except Exception as e:
        raise ValueError(f"Configuration validation failed: {e}")
