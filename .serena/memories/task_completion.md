# Task Completion Checklist

When finishing a task in this project:

- Ensure changes follow PEP 8 and project conventions (type hints, naming, logging via `logging`).
- Keep changes focused and small; avoid bundling refactors with feature changes unless necessary.
- If applicable, manually sanity-check behavior using `uv run uimm -- ...` or `uv run uimm-mcp` depending on which part you changed.
- Since there is no formal test suite, consider adding small ad-hoc scripts or using the Python REPL for edge-case checks around new logic.
- Update `README.md` or configuration docs if you introduce new environment variables, config fields, or CLI options.
- For commits/PRs, use clear imperative messages (e.g., `Add MCP client wrapper`, `Tune VAD parameters`) and note manual testing steps and any new config/env requirements.
