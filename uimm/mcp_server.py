from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

import httpx
from bs4 import BeautifulSoup
from mcp.server import NotificationOptions, Server
from mcp.server.stdio import stdio_server


UBTN_URL = "https://leiros.cloudfree.jp/usbtn/usbtn.html"
BASE_AUDIO_URL = "https://leiros.cloudfree.jp/usbtn/"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
HTML_CACHE_PATH = PROJECT_ROOT / ".uimm_cache" / "html" / "usbtn.html"


@dataclass
class AudioItem:
    id: str
    label: str
    audio_url: str
    volume: float | None
    category: str | None
    keywords: str | None
    video_id: str | None
    video_time: str | None


async def fetch_audio_items() -> List[AudioItem]:
    logger = logging.getLogger("uimm.mcp")

    # Ensure cache directory exists.
    HTML_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

    html: str
    if HTML_CACHE_PATH.is_file():
        try:
            html = HTML_CACHE_PATH.read_text(encoding="utf-8", errors="ignore")
            logger.info("Loaded USBTN page from cache: %s", HTML_CACHE_PATH)
        except OSError as exc:
            logger.warning("Failed to read USBTN cache %s: %s; refetching", HTML_CACHE_PATH, exc)
            async with httpx.AsyncClient(timeout=20) as client:
                resp = await client.get(UBTN_URL)
                resp.raise_for_status()
                html = resp.text
            try:
                HTML_CACHE_PATH.write_text(html, encoding="utf-8")
                logger.info("Cached USBTN page to %s", HTML_CACHE_PATH)
            except OSError as write_exc:  # pragma: no cover
                logger.warning("Failed to write USBTN cache %s: %s", HTML_CACHE_PATH, write_exc)
    else:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(UBTN_URL)
            resp.raise_for_status()
            html = resp.text
        try:
            HTML_CACHE_PATH.write_text(html, encoding="utf-8")
            logger.info("Cached USBTN page to %s", HTML_CACHE_PATH)
        except OSError as exc:  # pragma: no cover
            logger.warning("Failed to write USBTN cache %s: %s", HTML_CACHE_PATH, exc)

    # Fast path: directly locate the JS array
    marker = "let audioResourceList = ["
    start = html.find(marker)
    if start == -1:
        # Fallback: try to extract via BeautifulSoup, though unlikely needed
        soup = BeautifulSoup(html, "html.parser")
        scripts = "".join(s.get_text("\n") for s in soup.find_all("script"))
        html = scripts
        start = html.find(marker)
        if start == -1:
            raise RuntimeError("audioResourceList not found in HTML")

    start += len(marker)
    end = html.find("];", start)
    if end == -1:
        raise RuntimeError("audioResourceList terminator not found")

    array_text = html[start:end].strip()
    # Remove any commented-out lines (e.g., prototype entries), which break JSON parsing.
    lines: list[str] = []
    for line in array_text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("//"):
            continue
        lines.append(line)
    clean_array_text = "\n".join(lines)
    # The source already looks like JSON objects, so we mainly need [ and ].
    json_text = "[" + clean_array_text + "]"

    try:
        raw_items = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Failed to parse audioResourceList: {exc}") from exc

    items: List[AudioItem] = []
    for obj in raw_items:
        src = obj.get("src")
        if not src:
            continue
        if src.startswith("./"):
            audio_url = BASE_AUDIO_URL + src[2:]
        else:
            audio_url = src
        items.append(
            AudioItem(
                id=str(obj.get("id")),
                label=str(obj.get("label", obj.get("id", ""))),
                audio_url=audio_url,
                volume=float(obj.get("volume", 1.0)) if obj.get("volume") is not None else None,
                category=str(obj.get("a")) if obj.get("a") is not None else None,
                keywords=str(obj.get("k")) if obj.get("k") is not None else None,
                video_id=str(obj.get("videoId")) if obj.get("videoId") is not None else None,
                video_time=str(obj.get("time")) if obj.get("time") is not None else None,
            )
        )

    logger.info("Parsed %d audio items from USBTN page", len(items))
    return items


def to_dict(item: AudioItem) -> dict[str, Any]:
    return {
        "id": item.id,
        "label": item.label,
        "audio_url": item.audio_url,
        "volume": item.volume,
        "category": item.category,
        "keywords": item.keywords,
        "video_id": item.video_id,
        "video_time": item.video_time,
    }


server = Server("uimm")


@server.list_tools()
async def list_tools() -> list[dict[str, Any]]:
    return [
        {
            "name": "uimm.list_uimm_audio_files",
            "description": "List funny Shigure Ui button audio clips. Optionally filter by a short natural-language query.",
            "inputSchema": {
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
        {
            "name": "uimm.get_uimm_audio_file",
            "description": "Get a single audio clip by its id.",
            "inputSchema": {
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
        {
            "name": "uimm.pick_uimm_audio",
            "description": "Pick a funny audio clip for the user given mood and situation.",
            "inputSchema": {
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
    ]


async def tool_list_uimm_audio_files(query: str | None = None) -> list[dict[str, Any]]:
    items = await fetch_audio_items()
    if query:
        q = query.lower()
        filtered: list[AudioItem] = []
        for item in items:
            haystack = " ".join(
                filter(
                    None,
                    [
                        item.label,
                        item.keywords,
                        item.category,
                    ],
                )
            ).lower()
            if q in haystack:
                filtered.append(item)
        items = filtered
    logging.getLogger("uimm.mcp").info(
        "list_uimm_audio_files: query=%r -> %d item(s)", query, len(items)
    )
    return [to_dict(i) for i in items]


async def tool_get_uimm_audio_file(audio_id: str) -> dict[str, Any] | None:
    items = await fetch_audio_items()
    for item in items:
        if item.id == audio_id:
            logging.getLogger("uimm.mcp").info("get_uimm_audio_file: found id=%s", audio_id)
            return to_dict(item)
    logging.getLogger("uimm.mcp").warning("get_uimm_audio_file: id not found: %s", audio_id)
    return None


def classify_tags(item: AudioItem) -> set[str]:
    text = " ".join(filter(None, [item.label, item.keywords])).lower()
    tags: set[str] = set()
    # Rough, heuristic tag mapping based on Japanese text and punctuation.
    mapping = [
        ("起きろ", "alarm"),
        ("おきろ", "alarm"),
        ("寝", "sleep"),
        ("デブ", "insult"),
        ("ブタ", "insult"),
        ("怒", "angry"),
        ("www", "funny"),
        ("！", "intense"),
    ]
    for key, tag in mapping:
        if key.lower() in text:
            tags.add(tag)
    if not tags:
        tags.add("generic")
    return tags


async def tool_pick_uimm_audio(mood: str | None, situation: str, intensity: int | None) -> dict[str, Any]:
    import random

    items = await fetch_audio_items()
    scored: list[tuple[float, AudioItem]] = []
    mood = (mood or "").lower()
    intensity = intensity or 2

    for item in items:
        score = random.random()  # base randomness so results vary
        tags = classify_tags(item)

        if mood:
            if mood in ("alarm", "wake", "wake-up") and "alarm" in tags:
                score += 2.0
            if mood in ("teasing", "insult", "bully") and "insult" in tags:
                score += 2.0
            if mood in ("chaotic", "funny") and ("funny" in tags or "intense" in tags):
                score += 1.5

        label = item.label or ""
        if intensity >= 2 and "！" in label:
            score += 0.5
        if intensity == 3 and len(label) > 8:
            score += 0.5

        scored.append((score, item))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = [i for _, i in scored[:5]]
    if not top:
        raise RuntimeError("No audio items available")

    primary = top[0]
    alternates = [to_dict(i) for i in top[1:]]
    logging.getLogger("uimm.mcp").info(
        "pick_uimm_audio: mood=%r intensity=%r -> primary id=%s label=%s",
        mood,
        intensity,
        primary.id,
        primary.label,
    )

    return {
        "primary": to_dict(primary),
        "alternates": alternates,
        "situation": situation,
        "mood": mood,
        "intensity": intensity,
    }


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any] | None) -> Any:
    arguments = arguments or {}
    if name == "uimm.list_uimm_audio_files":
        return await tool_list_uimm_audio_files(arguments.get("query"))
    if name == "uimm.get_uimm_audio_file":
        return await tool_get_uimm_audio_file(arguments.get("id", ""))
    if name == "uimm.pick_uimm_audio":
        return await tool_pick_uimm_audio(
            arguments.get("mood"),
            arguments.get("situation", ""),
            arguments.get("intensity"),
        )
    raise ValueError(f"Unknown tool: {name}")


async def run_server() -> None:
    async with stdio_server() as (read_stream, write_stream):
        init_opts = server.create_initialization_options(
            notification_options=NotificationOptions(),
            experimental_capabilities={},
        )
        await server.run(read_stream, write_stream, init_opts)


def main() -> None:
    asyncio.run(run_server())


if __name__ == "__main__":
    main()
