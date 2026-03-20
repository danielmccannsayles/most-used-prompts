import json
import tempfile
from pathlib import Path

from prompt_miner.parse import Prompt, extract_all, find_jsonl_files, parse_file


def _make_record(content, **overrides):
    """Create a minimal JSONL user record."""
    rec = {
        "type": "user",
        "uuid": overrides.get("uuid", "abc-123"),
        "timestamp": "2026-03-01T12:00:00Z",
        "sessionId": "sess-1",
        "cwd": "/tmp/test",
        "version": "2.1.0",
        "gitBranch": "main",
        "parentUuid": None,
        "message": {"role": "user", "content": content},
    }
    rec.update(overrides)
    return rec


def test_parse_file_extracts_user_prompts(tmp_path):
    f = tmp_path / "projects" / "test-project" / "sess1" / "sess1.jsonl"
    f.parent.mkdir(parents=True)
    records = [
        _make_record("Hello world", uuid="u1"),
        {"type": "assistant", "uuid": "u2", "message": {"role": "assistant", "content": []}},
        _make_record("Fix the bug in auth", uuid="u3"),
    ]
    f.write_text("\n".join(json.dumps(r) for r in records))

    prompts = list(parse_file(f, project="test-project"))
    assert len(prompts) == 2
    assert prompts[0].content == "Hello world"
    assert prompts[1].content == "Fix the bug in auth"
    assert prompts[0].uuid == "u1"


def test_skips_tool_results(tmp_path):
    f = tmp_path / "test.jsonl"
    f.parent.mkdir(parents=True, exist_ok=True)
    # Tool result has content as a list
    rec = _make_record([{"type": "tool_result", "content": "output"}])
    f.write_text(json.dumps(rec))

    prompts = list(parse_file(f))
    assert len(prompts) == 0


def test_skips_short_prompts(tmp_path):
    f = tmp_path / "test.jsonl"
    f.parent.mkdir(parents=True, exist_ok=True)
    rec = _make_record("ok")
    f.write_text(json.dumps(rec))

    prompts = list(parse_file(f))
    assert len(prompts) == 0


def test_skips_meta_messages(tmp_path):
    f = tmp_path / "test.jsonl"
    f.parent.mkdir(parents=True, exist_ok=True)
    rec = _make_record("Hello world", isMeta=True)
    f.write_text(json.dumps(rec))

    prompts = list(parse_file(f))
    assert len(prompts) == 0


def test_skips_command_prompts(tmp_path):
    f = tmp_path / "test.jsonl"
    f.parent.mkdir(parents=True, exist_ok=True)
    rec = _make_record("run <command-name>foo</command-name>")
    f.write_text(json.dumps(rec))

    prompts = list(parse_file(f))
    assert len(prompts) == 0


def test_find_jsonl_files_skips_subagents(tmp_path):
    main = tmp_path / "projects" / "proj" / "sess" / "sess.jsonl"
    sub = tmp_path / "projects" / "proj" / "sess" / "subagents" / "agent.jsonl"
    main.parent.mkdir(parents=True)
    sub.parent.mkdir(parents=True)
    main.write_text("{}")
    sub.write_text("{}")

    files = find_jsonl_files(tmp_path / "projects")
    assert len(files) == 1
    assert "subagents" not in str(files[0])


def test_extract_all(tmp_path):
    base = tmp_path / "projects" / "my-proj" / "sess1" / "sess1.jsonl"
    base.parent.mkdir(parents=True)
    records = [
        _make_record("First prompt", uuid="u1"),
        _make_record("Second prompt", uuid="u2"),
    ]
    base.write_text("\n".join(json.dumps(r) for r in records))

    prompts = list(extract_all(tmp_path / "projects"))
    assert len(prompts) == 2
