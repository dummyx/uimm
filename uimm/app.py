from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import queue
import threading
import time
import warnings
from dataclasses import dataclass
from typing import Any, List

import numpy as np
import sounddevice as sd

# Some dependencies still use pkg_resources, which is deprecated and emits a noisy
# UserWarning on recent Setuptools. Suppress that specific warning.
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="pkg_resources is deprecated as an API.*",
)

import librosa
import webrtcvad
from faster_whisper import WhisperModel
from openai import OpenAI

from .config import AudioConfig, FunConfig, LLMConfig, load_audio_config, load_fun_config, load_llm_config, load_toml_config
from .mcp_client import MCPAudioClient
from .mcp_server import tool_pick_uimm_audio
from .player import AudioPlayer


@dataclass
class Utterance:
    audio: np.ndarray
    transcript: str | None = None


logger = logging.getLogger("uimm.app")


class VADSegmenter:
    def __init__(self, cfg: AudioConfig) -> None:
        self.cfg = cfg
        self.vad = webrtcvad.Vad(cfg.vad_aggressiveness)
        self.frame_bytes = int(cfg.sample_rate * cfg.frame_duration_ms / 1000) * 2  # 16-bit mono
        # Heuristics: require that some fraction of frames in a segment are
        # classified as speech to avoid triggering on background noise.
        # 0.4 is a compromise: lower values allow more segments through,
        # higher values are stricter but can miss quiet speech.
        self.min_speech_ratio = 0.4

    def _frame_generator(self, audio: bytes) -> List[bytes]:
        n = self.frame_bytes
        return [audio[i : i + n] for i in range(0, len(audio), n)]

    def segment(self, audio: bytes) -> List[bytes]:
        frames = self._frame_generator(audio)
        speech_flags = [self.vad.is_speech(f, self.cfg.sample_rate) for f in frames if len(f) == self.frame_bytes]
        segments: List[bytes] = []
        current: list[bytes] = []
        current_flags: list[bool] = []
        in_speech = False
        silence_ms = 0

        def maybe_commit_segment() -> None:
            nonlocal current, current_flags, segments
            if not current:
                return
            audio_bytes = b"".join(current)
            duration_ms = len(audio_bytes) / 2 / self.cfg.sample_rate * 1000
            if duration_ms < self.cfg.min_utterance_ms:
                current = []
                current_flags = []
                return
            speech_frames = sum(1 for f in current_flags if f)
            total_frames = len(current_flags) or 1
            ratio = speech_frames / total_frames
            if ratio >= self.min_speech_ratio:
                logger.debug(
                    "Accepting segment: duration=%.1f ms, speech_ratio=%.2f (threshold=%.2f)",
                    duration_ms,
                    ratio,
                    self.min_speech_ratio,
                )
                segments.append(audio_bytes)
            else:
                logger.debug(
                    "Dropping segment: duration=%.1f ms, speech_ratio=%.2f (threshold=%.2f)",
                    duration_ms,
                    ratio,
                    self.min_speech_ratio,
                )
            current = []
            current_flags = []

        for flag, frame in zip(speech_flags, frames):
            if flag:
                current.append(frame)
                current_flags.append(True)
                in_speech = True
                silence_ms = 0
            else:
                if in_speech:
                    silence_ms += self.cfg.frame_duration_ms
                    current.append(frame)
                    current_flags.append(False)
                    if silence_ms >= self.cfg.silence_duration_ms:
                        maybe_commit_segment()
                        in_speech = False
                        silence_ms = 0

        if current:
            maybe_commit_segment()

        if segments:
            logger.debug("VAD produced %d segment(s)", len(segments))

        return segments


class MicListener:
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
            self.stream.stop()
            self.stream.close()
            self.stream = None
        logger.info("MicListener stopped")


class STTWorker(threading.Thread):
    def __init__(self, cfg: AudioConfig, segment_queue: "queue.Queue[bytes]", utterance_queue: "queue.Queue[Utterance]") -> None:
        super().__init__(daemon=True)
        self.cfg = cfg
        self.segment_queue = segment_queue
        self.utterance_queue = utterance_queue
        # Model size chosen for a balance of speed and quality.
        logger.info("Loading Faster-Whisper model '%s' (device=auto)...", self.cfg.stt_model)
        self.model = WhisperModel(self.cfg.stt_model, device="auto")
        logger.info("Faster-Whisper model loaded")

    def run(self) -> None:  # type: ignore[override]
        while True:
            segment = self.segment_queue.get()
            if segment is None:  # type: ignore[comparison-overlap]
                break
            audio_np = np.frombuffer(segment, dtype=np.int16).astype(np.float32) / 32768.0
            segments, _ = self.model.transcribe(audio_np, language="ja", beam_size=5)
            text_parts: list[str] = []
            for seg in segments:
                text_parts.append(seg.text)
            transcript = " ".join(t.strip() for t in text_parts).strip()
            if transcript:
                self.utterance_queue.put(Utterance(audio=audio_np, transcript=transcript))
                logger.info("Transcribed utterance: %s", transcript)


class LLMChat:
    def __init__(self, llm_cfg: LLMConfig, fun_cfg: FunConfig, mcp_client: MCPAudioClient, player: AudioPlayer) -> None:
        client_kwargs: dict[str, Any] = {"api_key": llm_cfg.api_key}
        if llm_cfg.base_url:
            client_kwargs["base_url"] = llm_cfg.base_url
        self.client = OpenAI(**client_kwargs)
        self.model = llm_cfg.model
        self.fun_cfg = fun_cfg
        self.mcp_client = mcp_client
        self.player = player
        self.last_sound_time = 0.0
        self.messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.build_system_prompt()},
        ]
        self.tools: list[dict[str, Any]] = [
            {
                "type": "function",
                "function": {
                    "name": "uimm.list_uimm_audio_files",
                    "description": "List Shigure Ui button audio clips. Optionally filter by a short natural-language query.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Optional search query describing what kind of clip to find (e.g., 'alarm', 'insult', 'congrats').",
                            }
                        },
                        "required": [],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "uimm.get_uimm_audio_file",
                    "description": "Get a single audio clip by its id.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "id": {
                                "type": "string",
                                "description": "Exact id of the audio clip.",
                            }
                        },
                        "required": ["id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "uimm.pick_uimm_audio",
                    "description": (
                        "Pick a funny Shigure Ui button audio clip for the user given mood and situation. "
                        "Use this when you want to actually play a sound effect."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "mood": {
                                "type": "string",
                                "description": "High-level mood like 'teasing', 'supportive', 'chaotic', 'alarm'.",
                            },
                            "situation": {
                                "type": "string",
                                "description": "Short description of what the user just said or is doing.",
                            },
                            "intensity": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 3,
                                "description": "How strong or extreme the clip should be (1-3).",
                            },
                        },
                        "required": ["situation"],
                    },
                },
            },
        ]

    def should_trigger_sound(self) -> bool:
        now = time.time()
        if now - self.last_sound_time < self.fun_cfg.cooldown_seconds:
            return False
        import random

        if random.random() < self.fun_cfg.chaos_level:
            self.last_sound_time = now
            return True
        return False

    def build_system_prompt(self) -> str:
        return (
            "You are a playful, slightly teasing chat companion that reacts with Shigure Ui button sounds. "
            "You hear the user through a microphone; their words are transcribed for you. "
            "You have tools to browse and pick funny audio clips. "
            "Only trigger sounds when it will be funny or supportive, and prefer asking or hinting before overusing them. "
            "When you want to play a sound, call the uimm.pick_uimm_audio tool with a short, vivid description of the situation."
        )

    def handle_utterance(self, transcript: str) -> None:
        logger.info("Handling utterance: %s", transcript)
        # Append user message
        self.messages.append({"role": "user", "content": transcript})

        # Decide whether we want to force a sound effect for this turn.
        chaos_triggered = self.should_trigger_sound()
        if chaos_triggered:
            tool_choice_param: Any = {
                "type": "function",
                "function": {"name": "uimm.pick_uimm_audio"},
            }
            logger.debug("Chaos triggered; forcing uimm.pick_uimm_audio tool call for this utterance")
        else:
            tool_choice_param = "auto"

        # First completion: may or may not include tool calls (or must, if forced above).
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            tools=self.tools,
            tool_choice=tool_choice_param,
        )
        message = completion.choices[0].message
        tool_calls = getattr(message, "tool_calls", None) or []

        if tool_calls:
            logger.debug("Assistant requested %d tool call(s)", len(tool_calls))
            # Record the assistant message that requested tools.
            self.messages.append(message.model_dump(exclude_none=True))

            # Execute each tool call via MCP server.
            for tool_call in tool_calls:
                name = tool_call.function.name
                try:
                    arguments = json.loads(tool_call.function.arguments or "{}")
                except json.JSONDecodeError:
                    arguments = {}

                logger.info("Calling MCP tool %s with args=%s", name, arguments)
                result = self.mcp_client.call_tool(name, arguments)

                # Optionally trigger playback for pick_uimm_audio.
                if name == "uimm.pick_uimm_audio":
                    if result is None:
                        logger.warning(
                            "MCP tool %s returned no result; falling back to in-process picker", name
                        )
                        try:
                            result = asyncio.run(
                                tool_pick_uimm_audio(
                                    arguments.get("mood"),
                                    arguments.get("situation", ""),
                                    arguments.get("intensity"),
                                )
                            )
                        except Exception as exc:  # pragma: no cover
                            logger.error("In-process audio picker failed: %s", exc)
                            result = None

                    if isinstance(result, dict):
                        primary = result.get("primary") or {}
                        url = primary.get("audio_url")
                        volume = primary.get("volume")
                        if url and chaos_triggered:
                            logger.info("Playing clip: id=%s label=%s", primary.get("id"), primary.get("label"))
                            self.player.play_url(url, volume=volume)

                # Add tool result message back into conversation.
                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": name,
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )

            # Second completion: model sees tool results and responds.
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
            )
            message = completion.choices[0].message

        # Final assistant message (either from first or second completion).
        if message.content:
            self.messages.append({"role": "assistant", "content": message.content})
            logger.info("Assistant reply: %s", message.content)
            print(f"Assistant: {message.content}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Shigure Ui chat companion (always-listening, funny SFX).")
    parser.add_argument(
        "--config",
        type=str,
        help="Path to a TOML config file. If omitted, standard locations are searched.",
    )
    parser.add_argument("--device", type=int, help="Input device index for microphone (sounddevice).")
    parser.add_argument("--sample-rate", type=int, help="Sample rate for microphone audio (default 16000).")
    parser.add_argument(
        "--stt-model",
        type=str,
        help="Faster-Whisper model name or path (default 'medium').",
    )
    parser.add_argument(
        "--vad-aggressiveness",
        type=int,
        choices=[0, 1, 2, 3],
        help="WebRTC VAD aggressiveness (0=least, 3=most aggressive).",
    )
    parser.add_argument(
        "--chaos-level",
        type=float,
        help="How often to trigger sounds (0.0–1.0). Overrides UIMM_CHAOS_LEVEL.",
    )
    parser.add_argument(
        "--cooldown-seconds",
        type=float,
        help="Minimum seconds between sounds. Overrides UIMM_COOLDOWN_SECONDS.",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Override LLM model name (OPENAI_MODEL). Must support tools/function calling.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default INFO, or from UIMM_LOG_LEVEL).",
    )
    parser.add_argument(
        "--preload-audio",
        action="store_true",
        help="Download and cache all available UIMM audio clips on startup.",
    )
    args = parser.parse_args()

    # Logging setup
    env_level = os.getenv("UIMM_LOG_LEVEL", "INFO").upper()
    level_name = args.log_level or env_level
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    toml_cfg = load_toml_config(args.config)
    if "_config_path" in toml_cfg:
        logger.info("Loaded configuration from %s", toml_cfg["_config_path"])
    else:
        logger.info("No TOML configuration file found; using defaults/env/CLI")

    audio_section = (toml_cfg.get("audio") or {})
    audio_cfg = load_audio_config(toml_cfg)
    fun_cfg = load_fun_config(toml_cfg)
    llm_cfg = load_llm_config(toml_cfg)

    # Apply CLI overrides
    if args.sample_rate:
        audio_cfg.sample_rate = args.sample_rate
    if args.vad_aggressiveness is not None:
        audio_cfg.vad_aggressiveness = args.vad_aggressiveness
    if args.device is not None:
        audio_cfg.input_device = args.device

    if args.stt_model:
        audio_cfg.stt_model = args.stt_model

    if args.chaos_level is not None:
        fun_cfg.chaos_level = args.chaos_level
    if args.cooldown_seconds is not None:
        fun_cfg.cooldown_seconds = args.cooldown_seconds

    if args.model:
        llm_cfg.model = args.model

    # Ensure processing sample rate is valid for webrtcvad (supports only 8k/16k/32k/48k).
    supported_sample_rates = [8000, 16000, 32000, 48000]
    if audio_cfg.sample_rate not in supported_sample_rates:
        nearest = min(supported_sample_rates, key=lambda r: abs(r - audio_cfg.sample_rate))
        logger.warning(
            "Requested processing sample rate %d Hz is not in supported set %s; snapping to %d Hz",
            audio_cfg.sample_rate,
            supported_sample_rates,
            nearest,
        )
        audio_cfg.sample_rate = nearest

    # Detect the highest sample rate supported by the selected input device and capture at that rate.
    # We then downsample to audio_cfg.sample_rate (default 16 kHz) using librosa.
    if audio_cfg.input_sample_rate is None:
        candidate_srs = [96000, 88200, 48000, 44100, 32000, 24000, 22050, 16000, 8000]
        for sr in candidate_srs:
            try:
                sd.check_input_settings(device=audio_cfg.input_device, channels=1, samplerate=sr)
            except Exception:
                continue
            audio_cfg.input_sample_rate = sr
            logger.info(
                "Selected input sample rate %d Hz for device %s",
                audio_cfg.input_sample_rate,
                audio_cfg.input_device if audio_cfg.input_device is not None else "default",
            )
            break
        if audio_cfg.input_sample_rate is None:
            # Fallback to processing sample rate.
            audio_cfg.input_sample_rate = audio_cfg.sample_rate
            logger.info(
                "Falling back to input sample rate %d Hz (same as processing sample rate)",
                audio_cfg.input_sample_rate,
            )

    print("Starting chat companion...")
    print("Press Ctrl+C to exit.")

    # Queues between threads
    mic_queue: "queue.Queue[bytes]" = queue.Queue(maxsize=50)
    segment_queue: "queue.Queue[bytes]" = queue.Queue(maxsize=10)
    utterance_queue: "queue.Queue[Utterance]" = queue.Queue(maxsize=10)

    listener = MicListener(audio_cfg, mic_queue)
    segmenter = VADSegmenter(audio_cfg)
    stt_worker = STTWorker(audio_cfg, segment_queue, utterance_queue)
    mcp_client = MCPAudioClient()
    from pathlib import Path

    player = AudioPlayer(cache_dir=Path(audio_cfg.cache_dir) if audio_cfg.cache_dir else None)
    chat = LLMChat(llm_cfg, fun_cfg, mcp_client=mcp_client, player=player)

    # Optionally preload all audio into the cache.
    if audio_cfg.preload_all_audio or args.preload_audio:
        try:
            from .mcp_server import fetch_audio_items

            print("Preloading UIMM audio clips into cache (this may take a while)...")
            items = asyncio.run(fetch_audio_items())
            logger.info("Preloading %d audio clips into cache", len(items))
            for item in items:
                # Allow Ctrl+C during preloading without leaving the process in a bad state.
                if item.audio_url:
                    player.ensure_cached(item.audio_url)
        except KeyboardInterrupt:
            print("\nStopping during preload...")
            logger.info("KeyboardInterrupt received during preload; exiting.")
            return
        except Exception as exc:  # pragma: no cover
            logger.error("Failed to preload audio clips: %s", exc)

    try:
        try:
            listener.start()
        except sd.PortAudioError as exc:
            logger.error("Failed to open microphone stream: %s", exc)
            print(
                "Failed to open microphone input. "
                "Make sure an input device is available and, if needed, pass --device INDEX "
                "(you can list devices with: uv run python -- -c \"import sounddevice as sd; print(sd.query_devices())\")",
            )
            return
        stt_worker.start()

        buffer = b""
        input_sr = audio_cfg.input_sample_rate or audio_cfg.sample_rate
        target_sr = audio_cfg.sample_rate
        while True:
            chunk = mic_queue.get()
            # Convert input chunk to float32 and resample down to target_sr using librosa.
            if input_sr != target_sr:
                int_data = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    resampled = librosa.resample(int_data, orig_sr=input_sr, target_sr=target_sr)
                chunk_bytes = (np.clip(resampled, -1.0, 1.0) * 32768.0).astype(np.int16).tobytes()
            else:
                chunk_bytes = chunk

            buffer += chunk_bytes
            segments = segmenter.segment(buffer)
            if segments:
                for seg in segments:
                    segment_queue.put(seg)
                buffer = b""
            try:
                utterance = utterance_queue.get_nowait()
            except queue.Empty:
                continue
            print(f"User: {utterance.transcript}")
            chat.handle_utterance(utterance.transcript)
    except KeyboardInterrupt:
        print("\nStopping...")
        logger.info("KeyboardInterrupt received; shutting down.")
    finally:
        # Stop microphone and any ongoing playback, and signal the STT worker to exit.
        try:
            listener.stop()
        except Exception as exc:  # pragma: no cover
            logger.debug("Error while stopping microphone listener: %s", exc)

        try:
            sd.stop()
        except Exception as exc:  # pragma: no cover
            logger.debug("Error while stopping sounddevice playback: %s", exc)

        # Send sentinel to STT worker so it can finish cleanly.
        try:
            segment_queue.put_nowait(None)  # type: ignore[arg-type]
        except Exception:
            # If the queue is full or already shutting down, ignore.
            pass

        # Give the STT worker a brief chance to exit cleanly.
        try:
            stt_worker.join(timeout=2.0)
        except Exception as exc:  # pragma: no cover
            logger.debug("Error while joining STT worker: %s", exc)


if __name__ == "__main__":
    main()
