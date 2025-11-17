# handsfreed

`handsfreed` is a local, real-time speech-to-text daemon for Linux. It uses the `faster-whisper` library to provide high-quality, offline transcription.

This package is the daemon component of the [Handsfree](https://github.com/achyudh/handsfree) project. It is controlled by the [`handsfreectl`](https://crates.io/crates/handsfreectl) command-line tool.

## Installation

### Manual Installation

`handsfreed` requires `PortAudio`, which is a dependency of the `sounddevice` Python library. You must install the `PortAudio` library and its development headers using your system's package manager.

*   **Debian/Ubuntu:**
    ```bash
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

The `handsfreed` daemon is controlled by the `handsfreectl` command-line tool. You must install it separately.

*   You can download pre-compiled binaries from the [handsfreectl releases page](https://github.com/achyudh/handsfreectl/releases).

*   If you have the Cargo installed, you can install `handsfreectl` from Crates.io:
    ```bash
    cargo install handsfreectl
    ```

Once the dependencies are installed, you can install `handsfreed` using `pip`:

```bash
pip install handsfreed
```

### Nix Flake

If you use the [Nix package manager](https://nixos.org/) with flakes enabled, the [Handsfree flake](https://github.com/achyudh/handsfree) provides both `handsfreectl` and `handsfreed` packages along with a Home Manager module to configure and manage the `handsfreed` daemon as a systemd service.

For detailed instructions on how to add the flake to your system and configure the service, please refer to the Handsfree flake's **[README](https://github.com/achyudh/handsfreed/blob/main/README.md)**.

## Usage

`handsfreed` is designed to be run as a background service.

1.  **Create a configuration file:**
    Create a configuration file at `~/.config/handsfree/config.toml`. You can start with the [example configuration](https://github.com/achyudh/handsfreed/blob/main/example.config.toml).

2.  **Run the daemon:** The daemon will start listening for commands from `handsfreectl`.
    ```bash
    handsfreed
    ```

3.  **Control with `handsfreectl`:**
    Use the `handsfreectl` CLI to start/stop transcription and check the status of the daemon.

## Configuration

`handsfreed` is configured via a TOML file located at `~/.config/handsfree/config.toml`. The configuration allows you to set up your audio input, Whisper model, VAD parameters, and more.

For a full list of configuration options, please see the [example configuration file](https://github.com/achyudh/handsfreed/blob/main/example.config.toml).

## License

This project is licensed under the GNU General Public License v3.0.
