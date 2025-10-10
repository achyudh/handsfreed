"""Pipeline components for audio processing and transcription."""

import numpy as np
from dataclasses import dataclass

from .ipc_models import CliOutputMode


@dataclass
class TranscriptionTask:
    """Data structure for passing audio to the transcriber."""

    audio: np.ndarray
    output_mode: CliOutputMode
