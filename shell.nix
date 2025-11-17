# shell.nix
#
# Use this file with `nix-shell` to create a development environment
# for the 'handsfreed' project.

{ pkgs ? import <nixpkgs> { } }:

let

  name = "handsfreed-dev-${pythonVersion.version}";

  pythonVersion = pkgs.python313;

  pythonDependencies = ps:
    with ps; [
      # Dependencies from pyproject.toml
      toml # >=0.10
      sounddevice # >=0.4 (May need system libs like portaudio, see below)
      faster-whisper # >=1.0 (Check availability and version in your nixpkgs)
      numpy # >=1.2
      pydantic # >=2.0

      # Build dependencies
      pytest
      pytest-asyncio
      setuptools # >=45
      wheel
      build
      twine
    ];

  # Create the specific Python environment with the listed packages
  pythonEnv = pythonVersion.withPackages pythonDependencies;

in pkgs.mkShell {
  inherit name;

  # List of packages needed in the development environment's PATH.
  buildInputs = [
    # The Python environment itself, including all specified packages.
    pythonEnv

    # System libraries that might be needed by Python packages.
    # 'sounddevice' often requires PortAudio.
    pkgs.portaudio

    # You might need ALSA libraries on Linux as well/instead, depending
    # on how sounddevice/portaudio is configured or used.
    # pkgs.alsa-lib
  ];

  # Optional: Commands to run automatically when entering the shell.
  shellHook = ''
    echo "Entered ${name}"
    echo "Python version: $(python --version)"
    # You can add other setup commands here, like setting environment variables.
    # For example, if your project needs to be in the Python path:
    # export PYTHONPATH="$PWD:$PYTHONPATH"
    # echo "PYTHONPATH set to: $PYTHONPATH"
    echo "Run 'python' or 'pytest'..."
  '';
}
