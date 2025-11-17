uimm – Shigure Ui Chat Companion
================================

This is a small “funny app” that listens to your microphone, transcribes what you say with Faster-Whisper, and lets an LLM react by selecting and playing clips from the Shigure Ui button site.

The app also exposes an MCP server (`uimm`) that tools can use to list and pick audio clips from:

- https://leiros.cloudfree.jp/usbtn/usbtn.html


Running with uv
---------------

Install dependencies (creates a `.venv` managed by `uv`):

```bash
uv sync
```

Make sure you have a working microphone and speakers, plus any system libraries needed by PortAudio (for `sounddevice`) and by `faster-whisper` / `onnxruntime` on your platform.


Environment configuration
-------------------------

Set up an OpenAI-compatible endpoint:

- `OPENAI_API_KEY` – API key for your OpenAI-compatible server.
- `OPENAI_BASE_URL` – base URL if you are not using api.openai.com (optional).
- `OPENAI_MODEL` – model name to use (for example `gpt-4.1-mini`), must support tools / function calling.

Optional fun controls:

- `UIMM_CHAOS_LEVEL` – between `0.0` and `1.0`; higher means the assistant is more likely to fire sounds.
- `UIMM_COOLDOWN_SECONDS` – minimum number of seconds between sound plays.


Configuration via TOML
----------------------

You can also configure the app via a TOML file. The loader looks in this order (first existing file wins):

1. `--config /path/to/uimm.toml` (CLI flag)  
2. `$UIMM_CONFIG` (environment variable)  
3. `./uimm.toml`  
4. `./config.toml`  
5. `$XDG_CONFIG_HOME/uimm/config.toml` or `~/.config/uimm/config.toml`

Example `uimm.toml`:

```toml
[llm]
api_key = "sk-your-key"           # optional if OPENAI_API_KEY is set
base_url = "https://api.openai.com/v1"
model = "gpt-4.1-mini"

[audio]
sample_rate = 16000
stt_model = "medium"              # Faster-Whisper model name or local path
frame_duration_ms = 30
vad_aggressiveness = 2           # Silero VAD aggressiveness (0=most sensitive, 3=most strict)
min_utterance_ms = 600
max_utterance_ms = 15000
silence_duration_ms = 800
# input_device = 0               # microphone device index; omit to use system default

[fun]
chaos_level = 0.4                 # base value, overridden by UIMM_CHAOS_LEVEL or --chaos-level
cooldown_seconds = 10.0           # base value, overridden by UIMM_COOLDOWN_SECONDS or --cooldown-seconds
```

There is also a ready-made example file in the repo:

- `uimm.example.toml`

You can copy it to `uimm.toml` (or another supported location) and edit it.

```toml
[llm]
api_key = "sk-your-key"           # optional if OPENAI_API_KEY is set
base_url = "https://api.openai.com/v1"
model = "gpt-4.1-mini"

[audio]
sample_rate = 16000
stt_model = "medium"              # e.g. "tiny", "small", "medium", "large-v3" or local path
frame_duration_ms = 30
vad_aggressiveness = 2           # Silero VAD aggressiveness (0=most sensitive, 3=most strict)
min_utterance_ms = 600
max_utterance_ms = 15000
silence_duration_ms = 800
# input_device = 0               # microphone device index; omit to use system default
cache_dir = "./.uimm_cache/audio" # optional custom cache directory (default is project_root/.uimm_cache/audio)
preload_all_audio = false         # set true to download all clips on startup

[fun]
chaos_level = 0.4                 # base value, overridden by UIMM_CHAOS_LEVEL or --chaos-level
cooldown_seconds = 10.0           # base value, overridden by UIMM_COOLDOWN_SECONDS or --cooldown-seconds
```

Precedence rules:

- Defaults < TOML file < environment variables < CLI flags.


Starting the MCP server
-----------------------

Run the MCP server (stdio-based, server name `uimm`):

```bash
uv run uimm-mcp
```

This server exposes tools:

- `uimm.list_uimm_audio_files`
- `uimm.get_uimm_audio_file`
- `uimm.pick_uimm_audio`


Running the chat companion
--------------------------

Start the always-listening chat companion:

```bash
uv run uimm -- [options]
```

It will:

- Continuously listen to your microphone with VAD-based endpointing (no wake word).
- Transcribe utterances with `faster-whisper`.
- Send transcripts to the configured LLM with the MCP tools enabled.
- Occasionally (based on chaos level and cooldown) pick a funny Shigure Ui clip and play it back.

Press `Ctrl+C` to stop the app.


Useful CLI options
------------------

You can tweak behavior without changing environment variables:

- `--device INT` – input device index for the microphone (see `sounddevice.query_devices()` in Python to list).  
- `--sample-rate INT` – microphone sample rate (default 16000).  
- `--stt-model NAME_OR_PATH` – Faster-Whisper model name or path (default `medium`).  
- `--vad-aggressiveness {0,1,2,3}` – Silero VAD aggressiveness (0=most sensitive, 3=most strict).  
- `--chaos-level FLOAT` – how often to trigger sounds (0.0–1.0), overrides `UIMM_CHAOS_LEVEL`.  
- `--cooldown-seconds FLOAT` – minimum seconds between sounds, overrides `UIMM_COOLDOWN_SECONDS`.  
- `--model NAME` – override `OPENAI_MODEL` (must support tools/function calling).
- `--preload-audio` – download and cache all UIMM audio clips at startup.  
