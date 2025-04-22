"""Command-line entry point for handsfreed daemon."""

import sys
from .main import run


if __name__ == "__main__":
    sys.exit(run())
