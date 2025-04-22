import tomllib
import os
import getpass
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field, field_validator


def get_default_config_path() -> Path:
    return Path.home() / ".config" / "handsfree" / "config.toml"


def get_default_socket_path() -> Path:
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
    log_dir = Path.home() / ".local" / "state" / "handsfree"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / "handsfreed.log"


class VadConfig(BaseModel):
    threshold: float = 0.5
    min_speech_duration_ms: int = 250
    min_silence_duration_ms: int = 1500

    @field_validator("min_silence_duration_ms", "min_speech_duration_ms")
    @classmethod
    def check_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("Durations must be positive")
        return v


class WhisperConfig(BaseModel):
    model: str = "small.en"  # Default model
    device: str = "auto"
    compute_type: str = "auto"
    language: Optional[str] = None
    beam_size: int = 5
    vad_filter: bool = False

    @field_validator("model")
    @classmethod
    def check_model_not_empty(cls, v: str) -> str:
        if not v:
            raise ValueError("Whisper model identifier cannot be empty")
        return v

    @field_validator("beam_size")
    @classmethod
    def check_beam_size_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("Beam size must be positive")
        return v


class OutputConfig(BaseModel):
    keyboard_command: str
    clipboard_command: str

    @field_validator("keyboard_command", "clipboard_command")
    @classmethod
    def check_commands_not_empty(cls, v: str) -> str:
        if not v:
            raise ValueError("Output commands must not be empty")
        return v


class DaemonConfig(BaseModel):
    log_level: str = "INFO"
    log_file: Optional[Path] = None
    socket_path: Optional[Path] = None

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
    whisper: WhisperConfig
    vad: VadConfig = Field(default_factory=VadConfig)
    output: OutputConfig
    daemon: DaemonConfig = Field(default_factory=DaemonConfig)


def load_config(path: Optional[Path] = None) -> AppConfig:
    """Loads and validates the configuration from a TOML file."""
    if path is None:
        path = get_default_config_path()

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    try:
        with open(path, "rb") as f:
            config_data = tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise ValueError(f"Error decoding TOML file: {path}\n{e}") from e
    except OSError as e:
        raise OSError(f"Error reading file: {path}\n{e}") from e

    try:
        app_config = AppConfig(**config_data)
        return app_config
    except Exception as e:
        raise ValueError(f"Configuration validation failed: {e}")
