from __future__ import annotations

import logging
from typing import List

import numpy as np
import torch

from .config import AudioConfig


logger = logging.getLogger("uimm.endpointing")


class VADSegmenter:
    """Silero VAD-based utterance segmenter.

    Keeps the same public API as the original in-app implementation:
    - constructed from AudioConfig
    - exposes segment(audio_bytes) -> list[bytes]
    """

    def __init__(self, cfg: AudioConfig) -> None:
        self.cfg = cfg
        self.threshold = self._map_aggressiveness(cfg.vad_aggressiveness)
        self.sample_rate = cfg.sample_rate
        # Internal streaming buffer and cursor (in samples, relative to buffer start).
        self._buffer: bytes = b""
        self._processed_samples: int = 0
        torch.set_num_threads(1)
        logger.info("Loading Silero VAD model (sampling_rate=%d)...", self.sample_rate)
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=True,
        )
        (get_speech_timestamps, _, _, _, _) = utils
        self.model = model
        self.get_speech_timestamps = get_speech_timestamps
        logger.info("Silero VAD model loaded")

    @staticmethod
    def _map_aggressiveness(level: int) -> float:
        mapping = {
            0: 0.35,
            1: 0.5,
            2: 0.6,
            3: 0.75,
        }
        return mapping.get(level, 0.6)

    def segment(self, audio: bytes) -> List[bytes]:
        """Feed a new chunk of audio and return any completed speech segments.

        The segmenter keeps an internal buffer so it can detect segments that span
        multiple chunks. On each call, only newly finished segments are returned.
        """
        if not audio:
            return []

        # Append new chunk to internal buffer.
        self._buffer += audio
        if not self._buffer:
            return []

        audio_np = np.frombuffer(self._buffer, dtype=np.int16).astype(np.float32) / 32768.0
        speech_timestamps = self.get_speech_timestamps(
            audio_np,
            self.model,
            sampling_rate=self.sample_rate,
            threshold=self.threshold,
            min_speech_duration_ms=self.cfg.min_utterance_ms,
            min_silence_duration_ms=self.cfg.silence_duration_ms,
            return_seconds=False,
        )

        segments: List[bytes] = []
        max_end_sample = self._processed_samples

        for stamp in speech_timestamps:
            start_sample = int(stamp["start"])
            end_sample = int(stamp["end"])
            # Skip segments we've already emitted.
            if end_sample <= self._processed_samples:
                continue
            if start_sample < self._processed_samples:
                start_sample = self._processed_samples
            duration_ms = (end_sample - start_sample) / self.sample_rate * 1000.0
            if duration_ms < self.cfg.min_utterance_ms:
                continue
            if duration_ms > self.cfg.max_utterance_ms:
                max_samples = int(self.cfg.max_utterance_ms * self.sample_rate / 1000.0)
                end_sample = start_sample + max_samples
            start_byte = max(start_sample * 2, 0)
            end_byte = min(end_sample * 2, len(self._buffer))
            if end_byte <= start_byte:
                continue
            segments.append(self._buffer[start_byte:end_byte])
            if end_sample > max_end_sample:
                max_end_sample = end_sample

        # Update cursor to the end of the newest emitted segment.
        self._processed_samples = max_end_sample

        # Prevent unbounded growth of the internal buffer by trimming audio that
        # is safely older than the most recent processed position.
        total_samples = len(self._buffer) // 2
        if total_samples:
            max_buffer_ms = self.cfg.max_utterance_ms + self.cfg.silence_duration_ms + 500
            max_buffer_samples = int(max_buffer_ms * self.sample_rate / 1000.0)
            if total_samples > max_buffer_samples:
                drop_samples = total_samples - max_buffer_samples
                drop_bytes = drop_samples * 2
                self._buffer = self._buffer[drop_bytes:]
                # Adjust processed cursor to stay relative to new buffer start.
                self._processed_samples = max(0, self._processed_samples - drop_samples)

        if segments:
            logger.debug("Silero VAD produced %d segment(s)", len(segments))

        return segments
