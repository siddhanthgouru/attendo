"""
Vector store for face embeddings.

When PINECONE_API_KEY is set in .env, uses Pinecone.
Otherwise, falls back to an in-memory store with numpy cosine similarity.
This lets you develop and test locally without any API keys.
"""
import numpy as np

from app.config import settings


class LocalVectorStore:
    """In-memory vector store using numpy. Good enough for development."""

    def __init__(self):
        # Maps student_id (str) -> embedding (np.ndarray)
        self._vectors: dict[str, np.ndarray] = {}

    def store_embedding(self, student_id: str, embedding: np.ndarray) -> None:
        self._vectors[student_id] = embedding.astype(np.float32)

    def query_embedding(self, embedding: np.ndarray, top_k: int = 1) -> list[dict]:
        """
        Find the closest stored embeddings using cosine similarity.

        Returns:
            List of dicts with 'student_id' and 'score', sorted by score descending.
        """
        if not self._vectors:
            return []

        query = embedding.astype(np.float32)
        results = []
        for sid, stored in self._vectors.items():
            # Both vectors are L2-normalized, so dot product = cosine similarity
            score = float(np.dot(query, stored))
            results.append({"student_id": sid, "score": score})

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def delete_embedding(self, student_id: str) -> None:
        self._vectors.pop(student_id, None)


class PineconeVectorStore:
    """Production vector store backed by Pinecone."""

    def __init__(self):
        from pinecone import Pinecone

        self._pc = Pinecone(api_key=settings.pinecone_api_key)
        self._index = self._pc.Index(settings.pinecone_index_name)

    def store_embedding(self, student_id: str, embedding: np.ndarray) -> None:
        self._index.upsert(vectors=[{
            "id": student_id,
            "values": embedding.tolist(),
        }])

    def query_embedding(self, embedding: np.ndarray, top_k: int = 1) -> list[dict]:
        result = self._index.query(
            vector=embedding.tolist(),
            top_k=top_k,
            include_metadata=False,
        )
        return [
            {"student_id": match.id, "score": match.score}
            for match in result.matches
        ]

    def delete_embedding(self, student_id: str) -> None:
        self._index.delete(ids=[student_id])


def get_vector_store() -> LocalVectorStore | PineconeVectorStore:
    """Return the appropriate vector store based on config."""
    if settings.pinecone_api_key and settings.pinecone_api_key != "your_pinecone_api_key_here":
        return PineconeVectorStore()
    return LocalVectorStore()


# Singleton instance shared across the app
vector_store = get_vector_store()
