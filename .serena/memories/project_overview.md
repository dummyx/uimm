# Project Overview

- Purpose: Small “funny app” chat companion that listens to your microphone, transcribes speech using Faster-Whisper, and lets an LLM react by selecting and playing Shigure Ui button sounds. Also exposes a `uimm` MCP server that tools can use to list and pick audio clips from the Shigure Ui button site.
- Tech stack: Python 3.13+, `uv` for packaging/runtime, `faster-whisper` for STT, `sounddevice` + `webrtcvad` for audio/VAD, `simpleaudio` for playback, `httpx` + `beautifulsoup4` for scraping audio metadata, `openai` for LLM calls, `mcp` for MCP server and tools.
- Entry points: `uimm` CLI (chat companion) and `uimm-mcp` CLI (MCP server), defined in `pyproject.toml`.
- High-level architecture:
  - `uimm/app.py` – main CLI and orchestration (mic/VAD, STT, LLM, MCP tools, audio playback).
  - `uimm/mcp_server.py` – MCP stdio server exposing tools to list/get/pick UIMM audio files.
  - `uimm/audio_input.py`, `uimm/endpointing.py`, `uimm/stt.py` – audio capture, VAD-based endpointing, transcription.
  - `uimm/llm.py` – LLM client/wrapper logic.
  - `uimm/config.py` – configuration loading from TOML, env vars, CLI options.
  - `uimm/mcp_client.py` – MCP client wrapper for using the `uimm` server as a tool.
  - `uimm/player.py` – audio playback utilities.
- Config: Environment variables (`OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL`, `UIMM_CHAOS_LEVEL`, `UIMM_COOLDOWN_SECONDS`) and/or `uimm.toml`/`uimm.example.toml` with [llm], [audio], [fun] sections; precedence is defaults < TOML < env vars < CLI flags.
