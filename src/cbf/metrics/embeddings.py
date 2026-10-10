import numpy as np
from sklearn.metrics.pairwise import cosine_distances


def mean_pairwise_distance(embeddings: np.ndarray) -> float:
    """Mean cosine distance over unique pairs (higher = more spread out)."""
    if embeddings.shape[0] < 2:
        return 0.0
    distances = cosine_distances(embeddings)
    iu = np.triu_indices_from(distances, k=1)
    vals = distances[iu]
    return float(np.mean(vals))


def cluster_by_similarity(embeddings: np.ndarray, threshold: float = 0.75) -> list[list[int]]:
    """Greedy single-linkage clustering on cosine similarity >= threshold."""
    n = len(embeddings)
    if n == 0:
        return []
    visited = [False] * n
    clusters: list[list[int]] = []
    for i in range(n):
        if visited[i]:
            continue
        cluster = [i]
        visited[i] = True
        for j in range(i + 1, n):
            if not visited[j]:
                sim = float(np.dot(embeddings[i], embeddings[j]))
                if sim >= threshold:
                    cluster.append(j)
                    visited[j] = True
        clusters.append(cluster)
    return clusters


def embed_texts(texts: list[str], model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
