from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import av
import httpx
import numpy as np
import sounddevice as sd


UIMM_REFERER = "https://leiros.cloudfree.jp/usbtn/usbtn.html"


@dataclass
class AudioPlayer:
    """Simple audio player for UIMM clips with basic caching."""

    sample_rate: int = 44100
    cache_dir: Optional[Path] = None

    def __post_init__(self) -> None:
        self._logger = logging.getLogger("uimm.player")
        if self.cache_dir is None:
            # Default to a cache directory inside the project root, e.g.
            # <project_root>/.uimm_cache/audio
            project_root = Path(__file__).resolve().parent.parent
            self.cache_dir = project_root / ".uimm_cache" / "audio"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._logger.info("Using audio cache directory: %s", self.cache_dir)

    def _decode_mp3(self, path: Path) -> np.ndarray:
        self._logger.debug("Decoding MP3 file: %s", path)
        container = av.open(str(path))
        stream = next((s for s in container.streams if s.type == "audio"), None)
        if stream is None:
            return np.zeros(0, dtype=np.float32)

        frames: list[np.ndarray] = []
        for packet in container.demux(stream):
            for frame in packet.decode():
                arr = frame.to_ndarray()
                # arr: (channels, samples). Convert to mono.
                if arr.ndim == 2:
                    arr = arr.mean(axis=0)
                frames.append(arr.astype(np.float32))

        if not frames:
            self._logger.warning("Decoded MP3 but got no audio frames")
            return np.zeros(0, dtype=np.float32)

        audio = np.concatenate(frames)
        # Ensure within [-1, 1]
        return np.clip(audio, -1.0, 1.0)

    def _cache_path_for_url(self, url: str) -> Path:
        # Use the basename of the URL as the cache filename.
        basename = url.split("/")[-1].split("?")[0]
        return self.cache_dir / basename  # type: ignore[operator]

    def ensure_cached(self, url: str) -> Optional[Path]:
        """Ensure the given URL is downloaded into the cache and return its path."""
        cache_path = self._cache_path_for_url(url)
        if cache_path.exists():
            self._logger.debug("Cache hit for %s -> %s", url, cache_path)
            return cache_path

        headers = {"Referer": UIMM_REFERER}
        self._logger.info("Downloading audio clip: %s", url)
        with httpx.Client(timeout=20) as client:
            resp = client.get(url, headers=headers)
            resp.raise_for_status()
            data = resp.content

        # Write to a temporary file then atomically move into place.
        tmp_path = cache_path.with_suffix(cache_path.suffix + ".part")
        try:
            with tmp_path.open("wb") as f:
                f.write(data)
            tmp_path.replace(cache_path)
            self._logger.info("Cached audio clip to %s", cache_path)
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass

        return cache_path

    def play_url(self, url: str, volume: Optional[float] = None) -> None:
        cache_path = self.ensure_cached(url)
        if cache_path is None:
            return

        audio = self._decode_mp3(cache_path)
        if audio.size == 0:
            self._logger.warning("Audio clip from %s decoded to empty array; skipping playback", url)
            return

        if volume is not None:
            self._logger.debug("Applying volume multiplier: %s", volume)
            audio = np.clip(audio * float(volume), -1.0, 1.0)

        self._logger.info("Starting playback (samples=%d, sample_rate=%d)", audio.size, self.sample_rate)
        sd.play(audio, self.sample_rate)
        sd.wait()
        self._logger.info("Playback finished")
