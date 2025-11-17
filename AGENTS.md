# Repository Guidelines

This repository contains a small Python application and MCP server built with `uv`. It provides an always‑listening “chat companion” plus a `uimm` MCP server for Shigure Ui button sounds.

## Project Structure & Module Organization

- `pyproject.toml` – project metadata, dependencies, CLI entrypoints.
- `uimm/` – main package:
  - `app.py` – chat companion CLI, mic/VAD, STT, LLM + MCP orchestration.
  - `mcp_server.py` – MCP server (`uimm`) and tools.
  - `config.py`, `mcp_client.py`, `player.py` – configuration, MCP client wrapper, audio playback.
- `uimm.example.toml` – example configuration file.
- `README.md` – usage, configuration, and MCP docs.

Keep new modules under `uimm/` and group by responsibility (I/O, config, integration, etc.).

## Build, Test, and Development Commands

- Install deps and create venv:
  - `uv sync`
- Run chat companion:
  - `uv run uimm -- [options]`
- Run MCP server:
  - `uv run uimm-mcp`

There is currently no formal test suite; prefer small ad‑hoc scripts or REPL checks for new logic.

## Coding Style & Naming Conventions

- Use Python 3.13+ features and type hints everywhere.
- Follow PEP 8 style (4‑space indentation, snake_case for functions/variables, PascalCase for classes).
- Avoid one‑letter names and inline comments unless clarifying something non‑obvious.
- Keep logging via the standard `logging` module; no print statements outside CLI‑style output.

## Commit & Pull Request Guidelines

- Write clear, imperative commit messages (e.g., `Add MCP client wrapper`, `Fix VAD segmentation edge case`).
- Keep changes focused and small; separate refactors from feature changes when possible.
- PRs should:
  - Summarize the change and rationale.
  - Mention any new config/env requirements.
  - Include manual testing notes (commands run, scenarios checked).

## General Agent Behavior

- Always solve the problem unless it is genuinely impossible to do so.
- Do not hide problems; surface root causes clearly and address them directly.***
