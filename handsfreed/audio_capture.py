"""Audio capture module."""

import asyncio
import logging
import numpy as np
from typing import List, Optional

from .config import AppConfig
from .ipc_models import CliOutputMode
from .pipelines import TranscriptionTask

logger = logging.getLogger(__name__)

# Settings required by Whisper model
SAMPLE_RATE = 16000
NUM_CHANNELS = 1
AUDIO_FORMAT = "s16"  # Signed 16-bit

# Define a buffer size for reading from stdout
BUFFER_SIZE = 4096

# Conversion factor for s16 to float32
INT16_TO_FLOAT32 = 1.0 / 32768.0


class PWRecordCapture:
    """Captures audio using a pw-record subprocess."""

    def __init__(self, config: AppConfig, transcription_queue: asyncio.Queue):
        """Initialize audio capture.

        Args:
            config: The application configuration.
            transcription_queue: Queue to put final TranscriptionTask onto.
        """
        self.audio_config = config.audio
        self.transcription_queue = transcription_queue

        # Internal state
        self._process: Optional[asyncio.subprocess.Process] = None
        self._reader_task: Optional[asyncio.Task] = None
        self._audio_buffer: List[bytes] = []
        self._output_mode: Optional[CliOutputMode] = None
        self._is_running = False

    async def _read_audio_stream(self):
        """Reads audio data from the subprocess stdout."""
        if not self._process or not self._process.stdout:
            logger.error("Audio process or stdout not available for reading.")
            return

        logger.info("Audio reader task started.")
        try:
            while True:
                data = await self._process.stdout.read(BUFFER_SIZE)
                if not data:
                    logger.info("pw-record stdout stream ended.")
                    break
                self._audio_buffer.append(data)
        except asyncio.CancelledError:
            logger.info("Audio reader task cancelled.")
        except Exception as e:
            logger.exception(f"Error in audio reader task: {e}")
        finally:
            logger.info("Audio reader task finished.")

    async def start(self, output_mode: CliOutputMode) -> None:
        """Start audio capture.

        Args:
            output_mode: The output mode for the transcription.
        """
        if self._is_running:
            logger.warning("Audio capture is already running.")
            return

        logger.info(f"Starting audio capture with pw-record (Output: {output_mode.value})")
        self._is_running = True
        self._output_mode = output_mode
        self._audio_buffer.clear()

        # Construct pw-record command
        command = [
            "pw-record",
            f"--target={self.audio_config.target}",
            f"--rate={SAMPLE_RATE}",
            f"--format={AUDIO_FORMAT}",
            f"--channels={NUM_CHANNELS}",
            "-",
        ]

        try:
            self._process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            logger.info(f"Started pw-record process with PID: {self._process.pid}")

            # Start the task to read from stdout
            self._reader_task = asyncio.create_task(self._read_audio_stream())

        except FileNotFoundError:
            logger.error(
                "'pw-record' command not found. Please ensure PipeWire is installed."
            )
            self._is_running = False
            raise
        except Exception as e:
            logger.exception("Failed to start pw-record process.")
            self._is_running = False
            raise

    async def stop(self) -> None:
        """Stop audio capture and queue the result for transcription."""
        if not self._is_running:
            logger.warning("Audio capture is not running.")
            return

        logger.info("Stopping audio capture.")

        # Stop the reader task
        if self._reader_task and not self._reader_task.done():
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                logger.debug("Audio reader task successfully cancelled.")

        # Stop the process
        if self._process and self._process.returncode is None:
            try:
                self._process.terminate()
                await asyncio.wait_for(self._process.wait(), timeout=2.0)
                logger.info("pw-record process terminated.")
            except asyncio.TimeoutError:
                logger.warning("Timeout waiting for pw-record to terminate, killing.")
                self._process.kill()
            except Exception as e:
                logger.exception(f"Error stopping pw-record process: {e}")

        self._process = None
        self._reader_task = None

        if not self._audio_buffer or not self._output_mode:
            logger.warning("No audio was recorded or output mode not set, skipping transcription.")
            self._is_running = False
            return

        # Process the buffered audio
        logger.info(f"Processing {len(self._audio_buffer)} recorded audio chunks.")
        full_audio_bytes = b"".join(self._audio_buffer)

        # Convert raw s16 bytes to float32 numpy array
        try:
            audio_array = np.frombuffer(full_audio_bytes, dtype=np.int16).astype(np.float32) * INT16_TO_FLOAT32

            # Create and queue the transcription task
            task = TranscriptionTask(audio=audio_array, output_mode=self._output_mode)
            await self.transcription_queue.put(task)
            logger.info("Queued final audio segment for transcription.")

        except Exception as e:
            logger.exception("Error processing recorded audio.")

        # Reset state
        self._audio_buffer.clear()
        self._output_mode = None
        self._is_running = False
        logger.info("Audio capture stopped and buffer cleared.")
