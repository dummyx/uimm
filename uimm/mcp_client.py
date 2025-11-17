from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import anyio
import logging
import sys
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


logger = logging.getLogger("uimm.mcp_client")


@dataclass
class MCPAudioClient:
    """Thin wrapper around the uimm MCP server for audio tools."""

    # Use the same Python interpreter as the main process so that the
    # MCP server sees the same environment and installed packages.
    command: str = sys.executable
    # By default spawn the in-project MCP server module.
    args: List[str] = field(default_factory=lambda: ["-m", "uimm.mcp_server"])
    timeout_seconds: float = 10.0

    async def _call_tool_async(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> Any:
        params = StdioServerParameters(command=self.command, args=self.args)
        try:
            logger.debug("Starting MCP server process: %s %s", self.command, " ".join(self.args))
            async with stdio_client(params) as (read_stream, write_stream):
                session = ClientSession(read_stream, write_stream)
                async with anyio.fail_after(self.timeout_seconds):
                    await session.initialize()
                    # Populate tool schemas (useful for validation, though not strictly required here).
                    await session.list_tools()
                    result = await session.call_tool(name, arguments or {})
                return result.structuredContent
        except TimeoutError:
            logger.warning("MCP call to %s timed out after %.1f seconds", name, self.timeout_seconds)
            return None
        except Exception as exc:  # pragma: no cover
            logger.error("MCP call to %s failed: %s", name, exc)
            return None

    def call_tool(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> Any:
        async def runner() -> Any:
            return await self._call_tool_async(name, arguments)

        return anyio.run(runner)

    def list_uimm_audio_files(self, query: str | None = None) -> Any:
        args: Dict[str, Any] = {}
        if query:
            args["query"] = query
        return self.call_tool("uimm.list_uimm_audio_files", args)

    def get_uimm_audio_file(self, audio_id: str) -> Any:
        return self.call_tool("uimm.get_uimm_audio_file", {"id": audio_id})

    def pick_uimm_audio(self, mood: str | None, situation: str, intensity: int | None = None) -> Any:
        args: Dict[str, Any] = {"situation": situation}
        if mood:
            args["mood"] = mood
        if intensity is not None:
            args["intensity"] = intensity
        return self.call_tool("uimm.pick_uimm_audio", args)
