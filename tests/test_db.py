import numpy as np

from prompt_miner.db import PromptDB
from prompt_miner.parse import Prompt


def _make_prompt(**overrides):
    defaults = dict(
        uuid="test-uuid-1",
        session_id="sess-1",
        project="test-project",
        cwd="/tmp/test",
        git_branch="main",
        version="2.1.0",
        timestamp="2026-03-01T12:00:00Z",
        content="Fix the authentication bug",
        char_length=26,
        parent_uuid=None,
        source_file="/tmp/test.jsonl",
    )
    defaults.update(overrides)
    return Prompt(**defaults)


def test_ingest_and_query(tmp_path):
    db = PromptDB(tmp_path / "test.db")
    prompts = [
        _make_prompt(uuid="u1", content="Hello"),
        _make_prompt(uuid="u2", content="World"),
    ]
    db.ingest_prompts(prompts, "/tmp/test.jsonl", 1234.0)

    assert db.get_prompt_count() == 2
    all_prompts = db.get_all_prompts()
    assert len(all_prompts) == 2


def test_dedup_by_uuid(tmp_path):
    db = PromptDB(tmp_path / "test.db")
    p = _make_prompt(uuid="u1")
    db.ingest_prompts([p], "/tmp/test.jsonl", 1234.0)
    db.ingest_prompts([p], "/tmp/test.jsonl", 1235.0)

    assert db.get_prompt_count() == 1


def test_file_ingestion_tracking(tmp_path):
    db = PromptDB(tmp_path / "test.db")
    db.ingest_prompts([_make_prompt()], "/tmp/test.jsonl", 1234.0)

    assert db.is_file_ingested("/tmp/test.jsonl", 1234.0)
    assert not db.is_file_ingested("/tmp/test.jsonl", 9999.0)
    assert not db.is_file_ingested("/tmp/other.jsonl", 1234.0)


def test_embeddings_roundtrip(tmp_path):
    db = PromptDB(tmp_path / "test.db")
    db.ingest_prompts([_make_prompt(uuid="u1")], "/tmp/test.jsonl", 1234.0)

    prompts = db.get_all_prompts()
    pid = prompts[0]["id"]
    vec = np.random.randn(384).astype(np.float32)
    vec = vec / np.linalg.norm(vec)

    db.store_embeddings([pid], vec.reshape(1, -1), "test-model")

    ids, vectors = db.get_all_embeddings("test-model")
    assert len(ids) == 1
    assert ids[0] == pid
    np.testing.assert_allclose(vectors[0], vec, atol=1e-6)


def test_clusters_roundtrip(tmp_path):
    db = PromptDB(tmp_path / "test.db")
    db.ingest_prompts(
        [_make_prompt(uuid=f"u{i}") for i in range(5)],
        "/tmp/test.jsonl",
        1234.0,
    )
    prompts = db.get_all_prompts()
    ids = [p["id"] for p in prompts]
    labels = [0, 0, 1, 1, -1]

    db.store_clusters(ids, labels, "run1")
    db.store_cluster_labels({0: "auth bugs", 1: "tests"}, "run1")

    summary = db.get_cluster_summary()
    assert len(summary) == 2
    assert summary[0]["count"] == 2

    cluster_prompts = db.get_cluster_prompts(0)
    assert len(cluster_prompts) == 2


def test_search_text(tmp_path):
    db = PromptDB(tmp_path / "test.db")
    db.ingest_prompts(
        [
            _make_prompt(uuid="u1", content="Fix the login bug"),
            _make_prompt(uuid="u2", content="Add new feature"),
        ],
        "/tmp/test.jsonl",
        1234.0,
    )

    results = db.search_text("login")
    assert len(results) == 1
    assert "login" in results[0]["content"]


def test_stats(tmp_path):
    db = PromptDB(tmp_path / "test.db")
    db.ingest_prompts(
        [_make_prompt(uuid="u1"), _make_prompt(uuid="u2", session_id="sess-2")],
        "/tmp/test.jsonl",
        1234.0,
    )

    s = db.get_stats()
    assert s["prompts"] == 2
    assert s["sessions"] == 2
    assert s["projects"] == 1
