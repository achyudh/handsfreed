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
      numpy # >=1.26
      pydantic # >=2.0
      faster-whisper # >=1.0
      onnxruntime # >=1.18.0

      # Build dependencies
      pytest
      pytest-asyncio
      setuptools # >=45
      wheel
    ];

  # Create the specific Python environment with the listed packages
  pythonEnv = pythonVersion.withPackages pythonDependencies;

in pkgs.mkShell {
  inherit name;

  # List of packages needed in the development environment's PATH.
  buildInputs = [
    # The Python environment itself, including all specified packages.
    pythonEnv

    # You will need pipewire for pw-record.
    # pkgs.pipewire

    # You might need ALSA libraries on Linux as well.
    # pkgs.alsa-lib
  ];

  shellHook = ''
    python -m handsfreed
  '';
}
