from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .config import CLAUDE_PROJECTS_DIR, DEFAULT_EMBED_MODEL
from .db import PromptDB
from .parse import find_jsonl_files, parse_file

app = typer.Typer(help="Mine, cluster, and reuse your Claude Code prompts.")
console = Console()


@app.command()
def ingest(
    force: bool = typer.Option(False, help="Re-ingest all files"),
    no_embed: bool = typer.Option(False, "--no-embed", help="Skip embedding generation"),
    no_cluster: bool = typer.Option(False, "--no-cluster", help="Skip clustering"),
):
    """Parse Claude Code history, store prompts, generate embeddings, and cluster."""
    db = PromptDB()

    # Step 1: Parse and store prompts
    files = find_jsonl_files()
    new_prompts = 0
    skipped = 0

    with console.status("[bold green]Ingesting prompts...") as status:
        for path in files:
            mtime = path.stat().st_mtime
            file_str = str(path)
            if not force and db.is_file_ingested(file_str, mtime):
                skipped += 1
                continue
            prompts = list(parse_file(path))
            if prompts:
                db.ingest_prompts(prompts, file_str, mtime)
                new_prompts += len(prompts)

    total = db.get_prompt_count()
    console.print(f"[green]Ingested {new_prompts} new prompts ({skipped} files unchanged). {total} total.")

    # Step 2: Generate embeddings
    if not no_embed:
        missing = db.get_prompts_without_embeddings(DEFAULT_EMBED_MODEL)
        if missing:
            console.print(f"[blue]Embedding {len(missing)} prompts...")
            from .embed import Embedder
            embedder = Embedder()
            texts = [p["content"] for p in missing]
            ids = [p["id"] for p in missing]
            vectors = embedder.encode(texts)
            db.store_embeddings(ids, vectors, DEFAULT_EMBED_MODEL)
            console.print(f"[green]Embedded {len(missing)} prompts.")
        else:
            console.print("[dim]All prompts already embedded.")

    # Step 3: Cluster
    if not no_embed and not no_cluster:
        ids, vectors = db.get_all_embeddings(DEFAULT_EMBED_MODEL)
        if vectors is not None and len(ids) >= 10:
            console.print("[blue]Clustering...")
            from .cluster import cluster_prompts, get_run_id, label_clusters

            labels = cluster_prompts(vectors)
            run_id = get_run_id()
            db.store_clusters(ids, labels.tolist(), run_id)

            # Generate cluster labels
            all_prompts = db.get_all_prompts()
            prompts_by_id = {p["id"]: p["content"] for p in all_prompts}
            cluster_labels = label_clusters(ids, labels, prompts_by_id, vectors)
            db.store_cluster_labels(cluster_labels, run_id)

            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = int((labels == -1).sum())
            console.print(f"[green]Found {n_clusters} clusters ({n_noise} unclustered).")
        else:
            console.print("[dim]Not enough prompts to cluster (need >= 10).")

    db.close()


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    top_k: int = typer.Option(20, "-k", "--top-k", help="Number of results"),
    project: str | None = typer.Option(None, "-p", "--project", help="Filter by project"),
    threshold: float = typer.Option(0.3, "-t", "--threshold", help="Minimum similarity"),
):
    """Semantic search for similar prompts."""
    from .embed import Embedder

    db = PromptDB()
    ids, vectors = db.get_all_embeddings(DEFAULT_EMBED_MODEL)
    if vectors is None:
        console.print("[red]No embeddings found. Run `pm ingest` first.")
        raise typer.Exit(1)

    embedder = Embedder()
    results = embedder.search(query, ids, vectors, top_k=top_k * 3)

    table = Table(title="Search Results")
    table.add_column("#", style="dim", width=4)
    table.add_column("ID", width=6)
    table.add_column("Score", width=6)
    table.add_column("Prompt", max_width=60)
    table.add_column("Project", width=20)
    table.add_column("Date", width=12)

    shown = 0
    for pid, score in results:
        if score < threshold:
            break
        if shown >= top_k:
            break
        prompt = db.get_prompt_by_id(pid)
        if not prompt:
            continue
        if project and project.lower() not in prompt["project"].lower():
            continue
        shown += 1
        text = prompt["content"][:80].replace("\n", " ")
        date = prompt["timestamp"][:10] if prompt["timestamp"] else ""
        proj = prompt["project"].replace("-Users-dmccanns-Desktop-", "").replace("-", "/")
        table.add_row(str(shown), str(pid), f"{score:.2f}", text, proj, date)

    console.print(table)
    db.close()


@app.command()
def clusters(
    min_size: int = typer.Option(2, "-m", "--min-size", help="Minimum cluster size"),
):
    """List all prompt clusters with labels and counts."""
    db = PromptDB()
    summary = db.get_cluster_summary()

    table = Table(title="Prompt Clusters")
    table.add_column("Cluster", width=8)
    table.add_column("Count", width=6)
    table.add_column("Label", max_width=40)

    for row in summary:
        if row["count"] >= min_size:
            table.add_row(str(row["cluster_id"]), str(row["count"]), row["label"])

    console.print(table)
    db.close()


@app.command()
def cluster(
    cluster_id: int = typer.Argument(..., help="Cluster ID to show"),
):
    """Show all prompts in a specific cluster."""
    db = PromptDB()
    prompts = db.get_cluster_prompts(cluster_id)

    if not prompts:
        console.print(f"[red]No prompts found in cluster {cluster_id}.")
        raise typer.Exit(1)

    table = Table(title=f"Cluster {cluster_id} ({len(prompts)} prompts)")
    table.add_column("ID", width=6)
    table.add_column("Prompt", max_width=70)
    table.add_column("Project", width=20)
    table.add_column("Date", width=12)

    for p in prompts:
        text = p["content"][:100].replace("\n", " ")
        date = p["timestamp"][:10] if p["timestamp"] else ""
        proj = p["project"].replace("-Users-dmccanns-Desktop-", "").replace("-", "/")
        table.add_row(str(p["id"]), text, proj, date)

    console.print(table)
    db.close()


@app.command()
def history(
    project: str | None = typer.Option(None, "-p", "--project", help="Filter by project"),
    limit: int = typer.Option(50, "-n", "--limit", help="Number of prompts"),
    since: str | None = typer.Option(None, help="Show prompts since ISO date"),
):
    """Browse recent prompts chronologically."""
    db = PromptDB()
    prompts = db.get_history(project=project, limit=limit, since=since)

    table = Table(title="Prompt History")
    table.add_column("ID", width=6)
    table.add_column("Prompt", max_width=70)
    table.add_column("Project", width=20)
    table.add_column("Date", width=12)

    for p in prompts:
        text = p["content"][:100].replace("\n", " ")
        date = p["timestamp"][:10] if p["timestamp"] else ""
        proj = p["project"].replace("-Users-dmccanns-Desktop-", "").replace("-", "/")
        table.add_row(str(p["id"]), text, proj, date)

    console.print(table)
    db.close()


@app.command()
def stats():
    """Show summary statistics."""
    db = PromptDB()
    s = db.get_stats()

    console.print(f"[bold]Prompts:[/]     {s['prompts']}")
    console.print(f"[bold]Projects:[/]    {s['projects']}")
    console.print(f"[bold]Sessions:[/]    {s['sessions']}")
    console.print(f"[bold]Clusters:[/]    {s['clusters']}")
    console.print(f"[bold]Unclustered:[/] {s['unclustered']}")
    min_d = (s["min_date"] or "")[:10]
    max_d = (s["max_date"] or "")[:10]
    console.print(f"[bold]Date range:[/]  {min_d} to {max_d}")

    db.close()


@app.command()
def export(
    prompt_id: int = typer.Argument(..., help="Prompt ID to export"),
):
    """Print raw prompt text to stdout (for piping)."""
    db = PromptDB()
    prompt = db.get_prompt_by_id(prompt_id)
    db.close()

    if not prompt:
        print(f"Prompt {prompt_id} not found.", file=sys.stderr)
        raise typer.Exit(1)

    print(prompt["content"], end="")


@app.command()
def reuse(
    prompt_id: int = typer.Argument(..., help="Prompt ID to copy to clipboard"),
):
    """Copy a prompt to the clipboard."""
    db = PromptDB()
    prompt = db.get_prompt_by_id(prompt_id)
    db.close()

    if not prompt:
        print(f"Prompt {prompt_id} not found.", file=sys.stderr)
        raise typer.Exit(1)

    subprocess.run(["pbcopy"], input=prompt["content"].encode(), check=True)
    print(f"Copied prompt {prompt_id} to clipboard.", file=sys.stderr)


if __name__ == "__main__":
    app()
