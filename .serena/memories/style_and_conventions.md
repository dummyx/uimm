# Style and Conventions

- Language level: Python 3.13+ with type hints everywhere.
- General style: PEP 8 (4-space indentation, snake_case for functions/variables, PascalCase for classes).
- Avoid one-letter variable names and inline comments except to clarify non-obvious behavior.
- Logging: use the standard `logging` module; avoid `print` except for CLI-style user-facing messages (if present).
- Module organization: keep new modules under `uimm/` and group by responsibility (I/O, config, LLM/STT, audio, MCP integration).
- Configuration: prefer centralizing config parsing in `uimm/config.py` and reusing its types.
