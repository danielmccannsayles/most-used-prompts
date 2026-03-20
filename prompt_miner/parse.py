from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .config import CLAUDE_PROJECTS_DIR, MIN_PROMPT_LENGTH


@dataclass
class Prompt:
    uuid: str
    session_id: str
    project: str
    cwd: str
    git_branch: str | None
    version: str | None
    timestamp: str
    content: str
    char_length: int
    parent_uuid: str | None
    source_file: str


def find_jsonl_files(base_dir: Path | None = None) -> list[Path]:
    """Find all conversation JSONL files, skipping subagent files."""
    base = base_dir or CLAUDE_PROJECTS_DIR
    if not base.exists():
        return []
    files = []
    for path in base.rglob("*.jsonl"):
        # Skip subagent conversation files
        if "subagents" in path.parts:
            continue
        files.append(path)
    return sorted(files)


def _derive_project(jsonl_path: Path) -> str:
    """Derive the project name from the JSONL file path.

    Structure: ~/.claude/projects/<project-dir>/<session-uuid>/<session-uuid>.jsonl
    or:        ~/.claude/projects/<project-dir>/<session-uuid>.jsonl
    """
    # Walk up from the file to find the part right after "projects"
    parts = jsonl_path.parts
    try:
        idx = parts.index("projects")
        return parts[idx + 1]
    except (ValueError, IndexError):
        return "unknown"


def _should_skip(content: str) -> bool:
    """Check if a prompt should be filtered out."""
    if len(content) < MIN_PROMPT_LENGTH:
        return True
    if "<command-name>" in content or "<local-command" in content:
        return True
    return False


def parse_file(path: Path, project: str | None = None) -> Iterator[Prompt]:
    """Parse a single JSONL file and yield user prompts."""
    proj = project or _derive_project(path)
    source = str(path)

    with open(path, "r", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Must be a user message
            if record.get("type") != "user":
                continue
            if record.get("isMeta"):
                continue

            msg = record.get("message", {})
            if msg.get("role") != "user":
                continue

            content = msg.get("content")
            # Skip tool results (content is a list, not a string)
            if not isinstance(content, str):
                continue

            if _should_skip(content):
                continue

            yield Prompt(
                uuid=record.get("uuid", ""),
                session_id=record.get("sessionId", ""),
                project=proj,
                cwd=record.get("cwd", ""),
                git_branch=record.get("gitBranch"),
                version=record.get("version"),
                timestamp=record.get("timestamp", ""),
                content=content,
                char_length=len(content),
                parent_uuid=record.get("parentUuid"),
                source_file=source,
            )


def extract_all(base_dir: Path | None = None) -> Iterator[Prompt]:
    """Extract all user prompts from all Claude Code conversation files."""
    for path in find_jsonl_files(base_dir):
        yield from parse_file(path)
