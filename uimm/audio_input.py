from __future__ import annotations

import logging
import queue

import sounddevice as sd

from .config import AudioConfig


logger = logging.getLogger("uimm.audio_input")


class MicListener:
    """Simple microphone capture wrapper around sounddevice.InputStream."""

    def __init__(self, cfg: AudioConfig, out_queue: "queue.Queue[bytes]") -> None:
        self.cfg = cfg
        self.out_queue = out_queue
        self.stream: sd.InputStream | None = None
        self.running = False

    def start(self) -> None:
        if self.running:
            return
        self.running = True
        input_sr = self.cfg.input_sample_rate or self.cfg.sample_rate
        self.stream = sd.InputStream(
            device=self.cfg.input_device,
            channels=1,
            samplerate=input_sr,
            dtype="int16",
            blocksize=int(input_sr * self.cfg.frame_duration_ms / 1000),
            callback=self._callback,
        )
        self.stream.start()
        logger.info(
            "MicListener started (device=%s, sample_rate=%d)",
            self.cfg.input_device if self.cfg.input_device is not None else "default",
            self.cfg.sample_rate,
        )

    def _callback(self, indata, frames, time_info, status) -> None:  # type: ignore[override]
        if not self.running:
            return
        self.out_queue.put(bytes(indata.tobytes()))

    def stop(self) -> None:
        self.running = False
        if self.stream is not None:
            try:
                # Abort immediately; this is less likely to hang than a graceful stop
                # if the underlying PortAudio stream is in a bad state.
                self.stream.abort()
            except Exception:
                # Best-effort shutdown; ignore errors on abort.
                pass
            try:
                self.stream.close()
            except Exception:
                pass
            self.stream = None
        logger.info("MicListener stopped")
