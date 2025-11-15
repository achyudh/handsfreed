import os
from pathlib import Path
from unittest.mock import patch

import pytest
from handsfreed.config import (
    AppConfig,
    get_default_config_path,
    get_default_log_path,
    get_default_socket_path,
    load_config,
)

VALID_CONFIG = {
    "audio": {
        "input_gain": 1.5,
        "dc_offset_correction": True,
        "dc_offset_window_ms": 600,
    },
    "whisper": {
        "model": "base",
        "device": "cpu",
        "compute_type": "float32",
        "language": "en",
        "beam_size": 3,
        "cpu_threads": 4,
    },
    "vad": {
        "threshold": 0.6,
        "min_speech_duration_ms": 300,
        "min_silence_duration_ms": 2000,
    },
    "output": {
        "keyboard_command": "xdotool type --delay 0",
        "clipboard_command": "wl-copy",
    },
    "daemon": {
        "log_level": "DEBUG",
        "log_file": "/custom/log/path.log",
        "socket_path": "/custom/socket/path.sock",
    },
}


@pytest.fixture
def mock_config_file(tmp_path):
    """Creates a mock config file with valid TOML content."""
    config_file = tmp_path / "config.toml"
    config_content = """
        [audio]
        input_gain = 1.5
        dc_offset_correction = true
        dc_offset_window_ms = 600

        [whisper]
        model = "base"
        device = "cpu"
        compute_type = "float32"
        language = "en"
        beam_size = 3
        cpu_threads = 4

        [vad]
        threshold = 0.6
        min_speech_duration_ms = 300
        min_silence_duration_ms = 2000

        [output]
        keyboard_command = "xdotool type --delay 0"
        clipboard_command = "wl-copy"

        [daemon]
        log_level = "DEBUG"
        log_file = "/custom/log/path.log"
        socket_path = "/custom/socket/path.sock"
        """
    config_file.write_text(config_content)
    return config_file


def test_load_config_success(mock_config_file):
    """Test loading a valid configuration file."""
    config = load_config(mock_config_file)
    assert isinstance(config, AppConfig)
    assert config.audio.input_gain == 1.5
    assert config.audio.dc_offset_correction is True
    assert config.audio.dc_offset_window_ms == 600
    assert config.whisper.model == "base"
    assert config.whisper.device == "cpu"
    assert config.whisper.cpu_threads == 4
    assert config.vad.threshold == 0.6
    assert config.output.keyboard_command == "xdotool type --delay 0"
    assert config.daemon.log_level == "DEBUG"


def test_load_config_missing_uses_defaults():
    """Test loading with no config file uses defaults."""
    config = load_config(Path("/nonexistent/config.toml"))
    assert isinstance(config, AppConfig)
    assert config.audio.input_gain == 1.0
    assert config.audio.dc_offset_correction is True
    assert config.audio.dc_offset_window_ms == 512
    assert config.whisper.model == "small.en"  # Check default
    assert config.output.keyboard_command is None  # Check default is None now
    assert config.output.clipboard_command is None  # Check default is None now
    assert config.daemon.log_level == "INFO"  # Check default


def test_load_config_invalid_toml(tmp_path):
    """Test handling of invalid TOML syntax."""
    invalid_file = tmp_path / "invalid.toml"
    invalid_file.write_text("invalid { toml = syntax")
    with pytest.raises(ValueError, match="Error decoding TOML file"):
        load_config(invalid_file)


def test_audio_config_validation():
    """Test Audio configuration validation."""
    with pytest.raises(ValueError, match="greater than 0"):
        AppConfig(**{**VALID_CONFIG, "audio": {"input_gain": 0.0}})
    with pytest.raises(ValueError, match="greater than or equal to 0"):
        AppConfig(**{**VALID_CONFIG, "audio": {"dc_offset_window_ms": -1}})


def test_vad_config_validation():
    """Test VAD configuration validation."""
    with pytest.raises(ValueError, match="greater than or equal to 1"):
        AppConfig(
            **{**VALID_CONFIG, "vad": {"threshold": 0.5, "min_speech_duration_ms": -1}}
        )


def test_whisper_config_validation():
    """Test Whisper configuration validation."""
    with pytest.raises(ValueError):
        AppConfig(
            **{**VALID_CONFIG, "whisper": {**VALID_CONFIG["whisper"], "beam_size": 0}}
        )

    # Test cpu_threads validation
    with pytest.raises(ValueError):
        AppConfig(
            **{
                **VALID_CONFIG,
                "whisper": {**VALID_CONFIG["whisper"], "cpu_threads": -1},
            }
        )


def test_whisper_config_defaults():
    """Test Whisper configuration default values."""
    # Create config with minimal whisper settings
    config = AppConfig(**{**VALID_CONFIG, "whisper": {"model": "base"}})

    # Check defaults are applied correctly
    assert config.whisper.cpu_threads == 0


def test_output_config_validation():
    """Test output configuration validation."""
    # Since we've changed OutputConfig to allow None values,
    # this test is no longer relevant. Now we'll just verify
    # that empty strings are still valid.
    config = AppConfig(
        **{
            **VALID_CONFIG,
            "output": {"keyboard_command": "", "clipboard_command": "wl-copy"},
        }
    )
    assert config.output.keyboard_command == ""
    assert config.output.clipboard_command == "wl-copy"


def test_daemon_config_validation():
    """Test daemon configuration validation."""
    with pytest.raises(ValueError, match="Invalid log level"):
        AppConfig(
            **{
                **VALID_CONFIG,
                "daemon": {**VALID_CONFIG["daemon"], "log_level": "INVALID"},
            }
        )


def test_default_config_path():
    """Test config path follows XDG spec."""
    with patch.dict(os.environ, {"XDG_CONFIG_HOME": "/xdg/config"}, clear=True):
        expected = Path("/xdg/config/handsfree/config.toml")
        assert get_default_config_path() == expected

    # Test fallback to ~/.config
    with patch.dict(os.environ, {}, clear=True):
        expected = Path.home() / ".config" / "handsfree" / "config.toml"
        assert get_default_config_path() == expected


@pytest.mark.parametrize(
    "xdg_set,expected_base",
    [
        (True, "XDG_RUNTIME_DIR_VALUE/handsfree/daemon.sock"),
        (False, "/tmp/handsfree-testuser.sock"),
    ],
)
def test_default_socket_path(xdg_set, expected_base):
    """Test socket path defaults with and without XDG_RUNTIME_DIR."""
    with (
        patch.dict(
            os.environ,
            {"XDG_RUNTIME_DIR": "XDG_RUNTIME_DIR_VALUE"} if xdg_set else {},
            clear=True,
        ),
        patch("pathlib.Path.mkdir"),
        patch("os.access", return_value=True),
        patch("getpass.getuser", return_value="testuser"),
    ):
        path = get_default_socket_path()
        assert str(path) == expected_base


def test_default_log_path():
    """Test log path follows XDG spec."""
    with (
        patch.dict(os.environ, {"XDG_STATE_HOME": "/xdg/state"}, clear=True),
        patch("pathlib.Path.mkdir"),
    ):
        expected = Path("/xdg/state/handsfree/handsfreed.log")
        assert get_default_log_path() == expected

    # Test fallback to ~/.local/state
    with patch.dict(os.environ, {}, clear=True), patch("pathlib.Path.mkdir"):
        expected = Path.home() / ".local" / "state" / "handsfree" / "handsfreed.log"
        assert get_default_log_path() == expected


def test_computed_paths():
    """Test computed path properties."""
    config = AppConfig(**VALID_CONFIG)
    assert config.daemon.computed_log_file == Path(VALID_CONFIG["daemon"]["log_file"])
    assert config.daemon.computed_socket_path == Path(
        VALID_CONFIG["daemon"]["socket_path"]
    )

    # Test defaults when not specified
    minimal_config = AppConfig()  # All defaults
    assert minimal_config.daemon.computed_log_file == get_default_log_path()
    assert minimal_config.daemon.computed_socket_path == get_default_socket_path()
