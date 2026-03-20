from pathlib import Path

CLAUDE_PROJECTS_DIR = Path.home() / ".claude" / "projects"
DATA_DIR = Path.home() / ".prompt-miner"
DB_PATH = DATA_DIR / "prompts.db"

DEFAULT_EMBED_MODEL = "all-MiniLM-L6-v2"
EMBED_BATCH_SIZE = 256
MIN_PROMPT_LENGTH = 5
