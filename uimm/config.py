from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import tomllib


@dataclass
class LLMConfig:
    api_key: str
    base_url: str | None
    model: str


@dataclass
class AudioConfig:
    # Target processing sample rate (for VAD and STT).
    sample_rate: int = 16000
    # Name or path of the Faster-Whisper model to use.
    stt_model: str = "medium"
    frame_duration_ms: int = 30
    vad_aggressiveness: int = 2
    min_utterance_ms: int = 600
    max_utterance_ms: int = 15000
    silence_duration_ms: int = 800
    # Optional input device index and hardware sample rate.
    input_device: int | None = None
    input_sample_rate: int | None = None
    # Optional cache directory for downloaded audio.
    cache_dir: str | None = None
    # Whether to download all available audio clips on startup.
    preload_all_audio: bool = False


@dataclass
class FunConfig:
    chaos_level: float = 0.4
    cooldown_seconds: float = 10.0


def load_toml_config(explicit_path: str | None = None) -> dict[str, Any]:
    """Load configuration from a TOML file if present.

    Search order (first existing wins):
    1. explicit_path (CLI --config)
    2. $UIMM_CONFIG
    3. ./uimm.toml
    4. ./config.toml
    5. $XDG_CONFIG_HOME/uimm/config.toml or ~/.config/uimm/config.toml
    """
    candidates: list[Path] = []

    if explicit_path:
        candidates.append(Path(explicit_path))

    env_path = os.getenv("UIMM_CONFIG")
    if env_path:
        candidates.append(Path(env_path))

    candidates.append(Path("uimm.toml"))
    candidates.append(Path("config.toml"))

    xdg_config_home = os.getenv("XDG_CONFIG_HOME")
    if xdg_config_home:
        candidates.append(Path(xdg_config_home) / "uimm" / "config.toml")
    else:
        candidates.append(Path.home() / ".config" / "uimm" / "config.toml")

    for path in candidates:
        if path.is_file():
            with path.open("rb") as f:
                data = tomllib.load(f)
            # Stash the path in the config for logging / debugging.
            if "_config_path" not in data:
                data["_config_path"] = str(path)
            return data

    return {}


def load_llm_config(config: Mapping[str, Any] | None = None) -> LLMConfig:
    section = (config or {}).get("llm") or {}

    api_key = os.getenv("OPENAI_API_KEY") or section.get("api_key") or ""
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set and no [llm.api_key] in config file")

    base_url = os.getenv("OPENAI_BASE_URL") or section.get("base_url") or None
    model = os.getenv("OPENAI_MODEL") or section.get("model") or "gpt-4.1-mini"

    return LLMConfig(api_key=api_key, base_url=base_url, model=model)


def load_audio_config(config: Mapping[str, Any] | None = None) -> AudioConfig:
    section = (config or {}).get("audio") or {}

    return AudioConfig(
        sample_rate=int(section.get("sample_rate", AudioConfig.sample_rate)),
        stt_model=str(section.get("stt_model", AudioConfig.stt_model)),
        frame_duration_ms=int(section.get("frame_duration_ms", AudioConfig.frame_duration_ms)),
        vad_aggressiveness=int(section.get("vad_aggressiveness", AudioConfig.vad_aggressiveness)),
        min_utterance_ms=int(section.get("min_utterance_ms", AudioConfig.min_utterance_ms)),
        max_utterance_ms=int(section.get("max_utterance_ms", AudioConfig.max_utterance_ms)),
        silence_duration_ms=int(section.get("silence_duration_ms", AudioConfig.silence_duration_ms)),
        input_device=section.get("input_device"),
        input_sample_rate=section.get("input_sample_rate"),
        cache_dir=section.get("cache_dir"),
        preload_all_audio=bool(section.get("preload_all_audio", AudioConfig.preload_all_audio)),
    )


def load_fun_config(config: Mapping[str, Any] | None = None) -> FunConfig:
    section = (config or {}).get("fun") or {}

    # Start from file values, then let env vars override.
    chaos = section.get("chaos_level", 0.4)
    cooldown = section.get("cooldown_seconds", 10.0)

    try:
        chaos = float(os.getenv("UIMM_CHAOS_LEVEL", str(chaos)))
    except ValueError:
        chaos = float(chaos)

    try:
        cooldown = float(os.getenv("UIMM_COOLDOWN_SECONDS", str(cooldown)))
    except ValueError:
        cooldown = float(cooldown)

    return FunConfig(chaos_level=chaos, cooldown_seconds=cooldown)
