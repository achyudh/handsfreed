"""Audio capture module."""

import asyncio
import collections
import logging
import queue
import numpy as np
import sounddevice as sd
from typing import Optional

from .config import AudioConfig
from .pipeline import AbstractPipelineProducerComponent

logger = logging.getLogger(__name__)

# Settings required by Whisper model
SAMPLE_RATE = 16000
NUM_CHANNELS = 1
AUDIO_DTYPE = np.float32

# Small frames for segmentation strategies
FRAME_SIZE = 512  # frames (~32ms at 16kHz)

# Timeout for getting items from the thread-safe queue
QUEUE_GET_TIMEOUT = 0.1


class AudioCapture(AbstractPipelineProducerComponent):
    """Captures audio using sounddevice and provides raw audio frames."""

    def __init__(
        self,
        raw_audio_queue: asyncio.Queue,
        audio_config: AudioConfig,
        stop_event: asyncio.Event,
        device: Optional[int] = None,
    ):
        """Initialize audio capture.

        Args:
            raw_audio_queue: Queue to put raw audio frames onto
            audio_config: Audio configuration object.
            stop_event: Event to signal when to stop processing.
            device: Optional input device index (None for system default)
        """
        super().__init__(raw_audio_queue, stop_event)
        self.audio_config = audio_config
        self.device = device

        # Initialize internal state
        self._stream: Optional[sd.InputStream] = None
        self._raw_thread_q = queue.Queue()  # Thread-safe queue for callback data

        # Buffer for running DC offset calculation
        frame_duration_ms = (FRAME_SIZE / SAMPLE_RATE) * 1000
        dc_offset_buffer_maxlen = max(
            1, round(self.audio_config.dc_offset_window_ms / frame_duration_ms)
        )
        self._dc_offset_buffer: collections.deque = collections.deque(
            maxlen=dc_offset_buffer_maxlen
        )

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time,  # time info from sounddevice (unused)
        status: sd.CallbackFlags,
    ) -> None:
        """Callback for sounddevice's InputStream.

        Args:
            indata: Input audio data (frames x channels)
            frames: Number of frames in indata
            time: CData time info (unused)
            status: Status flags
        """
        if status:
            logger.warning(f"Audio callback status: {status}")

        try:
            # Ensure mono
            mono_data = indata[:, 0] if indata.shape[1] > 1 else indata.flatten()

            # Correct DC offset
            if self.audio_config.dc_offset_correction:
                self._dc_offset_buffer.append(np.mean(mono_data))
                running_offset = np.mean(self._dc_offset_buffer)
                mono_data = mono_data - running_offset

            # Apply input gain
            if self.audio_config.input_gain != 1.0:
                mono_data = mono_data * self.audio_config.input_gain

            # Put a copy onto the thread-safe queue
            self._raw_thread_q.put(mono_data.copy())
        except Exception as e:
            # Avoid crashes in audio callback
            logger.error(f"Error in audio callback: {e}")

    async def _produce_item(self) -> Optional[np.ndarray]:
        """Get a raw audio frame from the thread-safe queue."""
        try:
            return await asyncio.to_thread(
                self._raw_thread_q.get, timeout=QUEUE_GET_TIMEOUT
            )
        except queue.Empty:
            return None

    async def _on_start(self) -> None:
        """Start the sounddevice stream."""
        logger.info(f"Starting audio capture (Device: {self.device or 'Default'})")
        self._dc_offset_buffer.clear()

        # Clear the raw queue in case of restart
        while not self._raw_thread_q.empty():
            try:
                self._raw_thread_q.get_nowait()
            except queue.Empty:
                break

        try:
            # Start the sounddevice stream
            self._stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                device=self.device,
                channels=NUM_CHANNELS,
                dtype=AUDIO_DTYPE,
                callback=self._audio_callback,
                latency="low",
                blocksize=FRAME_SIZE,  # Small blocks for faster segmentation
            )

            # Run potentially blocking stream start in executor
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._stream.start)

            logger.info("Audio capture started")

        except sd.PortAudioError as e:
            logger.error(f"PortAudio error starting audio stream: {e}")
            self._stream = None
            raise
        except Exception as e:
            logger.exception(f"Failed to start audio capture: {e}")
            self._stream = None
            raise

    async def _on_stop(self) -> None:
        """Stop the sounddevice stream."""
        logger.info("Stopping audio capture...")

        # Stop and close the stream
        if self._stream is not None:
            try:
                # Run potentially blocking stream stop/close in executor
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, self._stream.stop)
                await loop.run_in_executor(None, self._stream.close)
                logger.info("Audio stream stopped and closed")
            except sd.PortAudioError as e:
                logger.error(f"PortAudio error stopping audio stream: {e}")
            except Exception as e:
                logger.exception(f"Error stopping audio stream: {e}")
            finally:
                self._stream = None

        logger.info("Audio capture stopped")
