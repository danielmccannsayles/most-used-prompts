from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone

import numpy as np


def cluster_prompts(
    vectors: np.ndarray,
    min_cluster_size: int = 5,
    min_samples: int = 3,
) -> np.ndarray:
    """Run HDBSCAN on embedding vectors. Returns array of cluster labels (-1 = noise)."""
    from sklearn.cluster import HDBSCAN

    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        n_jobs=-1,
    )
    clusterer.fit(vectors)
    return clusterer.labels_


def label_clusters(
    prompt_ids: list[int],
    labels: np.ndarray,
    prompts_by_id: dict[int, str],
    vectors: np.ndarray,
) -> dict[int, str]:
    """Generate a short label for each cluster from its most central prompts."""
    unique_labels = set(int(l) for l in labels)
    unique_labels.discard(-1)

    result = {}
    for cid in sorted(unique_labels):
        mask = labels == cid
        cluster_indices = np.where(mask)[0]
        cluster_vectors = vectors[cluster_indices]

        # Find centroid and the 3 prompts closest to it
        centroid = cluster_vectors.mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-10)
        dists = cluster_vectors @ centroid
        top_indices = np.argsort(dists)[::-1][:3]

        # Build label from the most common words in central prompts
        central_texts = []
        for idx in top_indices:
            pid = prompt_ids[cluster_indices[idx]]
            text = prompts_by_id.get(pid, "")
            central_texts.append(text)

        label = _extract_label(central_texts)
        result[cid] = label

    return result


def _extract_label(texts: list[str], max_words: int = 5) -> str:
    """Extract a short label from a set of texts using common words."""
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "and", "or", "but", "not", "this", "that", "it", "i", "me",
        "my", "we", "our", "you", "your", "can", "do", "does", "did",
        "will", "would", "should", "could", "have", "has", "had",
        "all", "if", "so", "no", "yes", "just", "also", "then",
        "what", "how", "when", "where", "which", "who", "why",
        "please", "make", "use", "like", "need", "want", "get",
    }

    words: list[str] = []
    for text in texts:
        # Take first 200 chars of each prompt
        text = text[:200].lower()
        for w in text.split():
            w = w.strip(".,!?\"'`()[]{}:;/\\<>")
            if len(w) > 2 and w not in stop_words and w.isalpha():
                words.append(w)

    counts = Counter(words)
    top = [w for w, _ in counts.most_common(max_words)]
    return " ".join(top) if top else "miscellaneous"


def get_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
