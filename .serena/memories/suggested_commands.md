# Suggested Commands

## Setup
- `uv sync` – create/refresh the project virtualenv and install dependencies.

## Run entrypoints
- `uv run uimm -- [options]` – start the always-listening chat companion (mic/VAD, STT, LLM + MCP tools, audio playback).
- `uv run uimm-mcp` – run the stdio-based MCP server (`uimm`) exposing tools for listing/getting/picking Shigure Ui audio clips.

## Development helpers
- There is currently no formal test suite; prefer small ad-hoc scripts or REPL checks when developing new logic.
- Use standard Linux tooling like `ls`, `rg`, `sed`, `python -m` invocations, etc., as needed during development.
