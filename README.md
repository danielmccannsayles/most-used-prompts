# prompt-miner

A local tool that mines your Claude Code conversation history, stores prompts in SQLite, generates semantic embeddings, and clusters similar prompts — helping you discover patterns and reuse your best prompts.

## Install

```bash
pip install -e .
```

## Usage

```bash
# Ingest prompts from ~/.claude/projects/, embed, and cluster
pm ingest

# Semantic search
pm search "review PR"

# Browse clusters
pm clusters
pm cluster 27

# Recent history
pm history
pm history -p pollux -n 20

# Stats
pm stats

# Reuse a prompt
pm export 42          # print to stdout
pm export 42 | claude # pipe into a new Claude session
pm reuse 42           # copy to clipboard
```

## How it works

1. **Parse** — Reads JSONL conversation files from `~/.claude/projects/`, extracts user prompts with metadata (project, session, timestamp, git branch)
2. **Store** — SQLite database at `~/.prompt-miner/prompts.db`, deduped by message UUID, incremental on re-runs
3. **Embed** — Local embeddings via `all-MiniLM-L6-v2` (sentence-transformers), ~384 dims, runs on CPU in seconds
4. **Cluster** — HDBSCAN groups semantically similar prompts, auto-generates labels from central prompts

## Commands

| Command             | Description                                                                            |
| ------------------- | -------------------------------------------------------------------------------------- |
| `pm ingest`         | Scan history, embed, cluster. `--force` to re-ingest all, `--no-embed`, `--no-cluster` |
| `pm search <query>` | Semantic search. `-k 10` top-k, `-t 0.5` threshold, `-p project` filter                |
| `pm clusters`       | List all clusters with counts and labels. `-m 3` min size                              |
| `pm cluster <id>`   | Show all prompts in a cluster                                                          |
| `pm history`        | Browse recent prompts. `-p project`, `-n limit`, `--since 2026-03-01`                  |
| `pm stats`          | Prompt count, projects, sessions, clusters, date range                                 |
| `pm export <id>`    | Raw prompt to stdout                                                                   |
| `pm reuse <id>`     | Copy prompt to clipboard                                                               |
