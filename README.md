# handsfreed

`handsfreed` is a local, real-time speech-to-text daemon for Linux. It uses the `faster-whisper` library to provide high-quality, offline transcription.

This package is the daemon component of the [Handsfree](https://github.com/achyudh/handsfree) project. It is controlled by the [`handsfreectl`](https://crates.io/crates/handsfreectl) command-line tool.

## Installation

### Manual Installation

`handsfreed` requires `PortAudio`, which is a dependency of the `sounddevice` Python library. You must install the `PortAudio` library and its development headers using your system's package manager.

*   **Debian/Ubuntu:**
    ```bash
    sudo apt-get update
    sudo apt-get install libportaudio2 libportaudiocpp0 portaudio19-dev
    ```

*   **Fedora/CentOS/RHEL:**
    ```bash
    sudo dnf install portaudio-devel
    ```

*   **Arch Linux:**
    ```bash
    sudo pacman -S portaudio
    ```

Once the system dependencies are installed, you can install `handsfreed` using `pip`:

```bash
pip install handsfreed
```

### Nix Flake

If you are using Nix with Flakes enabled, `handsfreed` can be installed by following the instructions [here](https://github.com/achyudh/handsfree#nix-flake).

## Usage

`handsfreed` is designed to be run as a background service.

1.  **Create a configuration file:**
    Create a configuration file at `~/.config/handsfree/config.toml`. You can start with the [example configuration](https://github.com/achyudh/handsfreed/blob/main/example.config.toml).

2.  **Run the daemon:**
    ```bash
    handsfreed
    ```
    The daemon will start listening for commands from `handsfreectl`.

3.  **Control with `handsfreectl`:**
    Use the [`handsfreectl`](https://github.com/achyudh/handsfreectl) CLI to start/stop transcription and check the status of the daemon. You must install it separately.

## Configuration

`handsfreed` is configured via a TOML file located at `~/.config/handsfree/config.toml`. The configuration allows you to set up your audio input, Whisper model, VAD parameters, and more.

For a full list of configuration options, please see the [example configuration file](https://github.com/achyudh/handsfreed/blob/main/example.config.toml).

## License

This project is licensed under the GNU General Public License v3.0.
