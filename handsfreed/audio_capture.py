"""Audio capture module."""

import asyncio
import logging
import queue
import numpy as np
import sounddevice as sd
from typing import Optional

logger = logging.getLogger(__name__)

# Settings required by Whisper model
SAMPLE_RATE = 16000
NUM_CHANNELS = 1
AUDIO_DTYPE = np.float32


class AudioCapture:
    """Captures audio using sounddevice and provides fixed-duration chunks."""

    def __init__(
        self,
        audio_queue: asyncio.Queue,
        chunk_duration_s: float = 5.0,
        device: Optional[int] = None,
        input_gain: float = 1.0,
    ):
        """Initialize audio capture.

        Args:
            audio_queue: Queue to put audio chunks onto
            chunk_duration_s: Duration of each chunk in seconds
            device: Optional input device index (None for system default)
            input_gain: Input gain multiplier (1.0 = unity gain)
        """
        self.audio_queue = audio_queue
        self.chunk_duration_s = chunk_duration_s
        self.device = device
        self.input_gain = input_gain

        # Calculate chunk size in frames
        self.chunk_size_frames = int(self.chunk_duration_s * SAMPLE_RATE)

        # Initialize internal state
        self._stream: Optional[sd.InputStream] = None
        self._buffer = np.array([], dtype=AUDIO_DTYPE)
        self._raw_audio_q = queue.Queue()  # Thread-safe queue for callback data
        self._processing_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()

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
            # Ensure mono and normalize
            mono_data = indata[:, 0] if indata.shape[1] > 1 else indata.flatten()

            # Apply input gain
            if self.input_gain != 1.0:
                mono_data = mono_data * self.input_gain

            # Put a copy onto the thread-safe queue
            self._raw_audio_q.put(mono_data.copy())
        except Exception as e:
            # Avoid crashes in audio callback
            logger.error(f"Error in audio callback: {e}")

    async def _process_new_data(self) -> bool:
        """Process new data from raw audio queue.

        Returns:
            True if data was processed, False if queue was empty.
        """
        try:
            raw_chunk = await asyncio.to_thread(self._raw_audio_q.get_nowait)
            self._buffer = np.concatenate((self._buffer, raw_chunk))
            return True
        except queue.Empty:
            return False

    async def _chunk_processor(self) -> None:
        """Process raw audio into fixed-duration chunks."""
        logger.info("Chunk processor task started")
        self._buffer = np.array([], dtype=AUDIO_DTYPE)

        while not self._stop_event.is_set():
            try:
                # Try to get and process new data
                got_data = await self._process_new_data()
                if not got_data:
                    # No new data, yield control
                    await asyncio.sleep(0.01)
                    continue

                # Process buffer if enough data for a chunk
                while len(self._buffer) >= self.chunk_size_frames:
                    # Extract chunk
                    chunk = self._buffer[: self.chunk_size_frames].astype(AUDIO_DTYPE)
                    # Remove chunk from buffer (non-overlapping for now)
                    self._buffer = self._buffer[self.chunk_size_frames :]

                    logger.debug(
                        f"Audio chunk ready: {len(chunk)} frames "
                        f"({len(chunk) / SAMPLE_RATE:.1f}s)"
                    )
                    await self.audio_queue.put(chunk)

            except asyncio.CancelledError:
                logger.info("Chunk processor task cancelled")
                break
            except Exception as e:
                logger.exception(f"Error in chunk processor task: {e}")
                # Avoid tight loop on error
                await asyncio.sleep(0.5)

        logger.info("Chunk processor task finished")

    async def start(self) -> None:
        """Start audio capture."""
        if self._stream is not None or self._processing_task is not None:
            logger.warning("Audio capture already running")
            return

        logger.info(
            f"Starting audio capture (Device: {self.device or 'Default'}, "
            f"Chunk Duration: {self.chunk_duration_s}s)"
        )
        self._stop_event.clear()

        # Clear the raw queue in case of restart
        while not self._raw_audio_q.empty():
            try:
                self._raw_audio_q.get_nowait()
            except queue.Empty:
                break

        self._buffer = np.array([], dtype=AUDIO_DTYPE)

        try:
            # Start the chunk processing task first
            self._processing_task = asyncio.create_task(self._chunk_processor())

            # Start the sounddevice stream
            self._stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                device=self.device,
                channels=NUM_CHANNELS,
                dtype=AUDIO_DTYPE,
                callback=self._audio_callback,
                latency="low",
            )

            # Run potentially blocking stream start in executor
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._stream.start)

            logger.info("Audio capture started")

        except sd.PortAudioError as e:
            logger.error(f"PortAudio error starting audio stream: {e}")
            if self._processing_task and not self._processing_task.done():
                self._processing_task.cancel()
            self._stream = None
            self._processing_task = None
            raise
        except Exception as e:
            logger.exception(f"Failed to start audio capture: {e}")
            if self._processing_task and not self._processing_task.done():
                self._processing_task.cancel()
            self._stream = None
            self._processing_task = None
            raise

    async def stop(self) -> None:
        """Stop audio capture."""
        if self._stream is None and self._processing_task is None:
            logger.warning("Audio capture not running")
            return

        logger.info("Stopping audio capture...")
        self._stop_event.set()  # Signal processor task to stop

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

        # Wait for the processing task to finish
        if self._processing_task is not None:
            try:
                # Wait with a timeout for the task to finish/cancel
                await asyncio.wait_for(self._processing_task, timeout=2.0)
                logger.info("Chunk processor task finished")
            except asyncio.TimeoutError:
                logger.warning("Timeout waiting for chunk processor task to finish")
                self._processing_task.cancel()
                # Wait a bit more after cancelling
                await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                logger.info("Chunk processor task was cancelled")
            except Exception as e:
                logger.exception(f"Error waiting for chunk processor task: {e}")
            finally:
                self._processing_task = None

        logger.info("Audio capture stopped")
