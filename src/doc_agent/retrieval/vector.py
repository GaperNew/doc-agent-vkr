from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from doc_agent.models import DocumentChunk, SearchResult
from doc_agent.utils.jsonl import read_jsonl


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class VectorRetriever:
    def __init__(
        self,
        chunks: list[DocumentChunk],
        embeddings: np.ndarray,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
    ) -> None:
        self.chunks = chunks
        self.embeddings = embeddings
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    @classmethod
    def from_chunks(
        cls,
        chunks: list[DocumentChunk],
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        batch_size: int = 32,
    ) -> "VectorRetriever":
        model = SentenceTransformer(model_name)

        texts = [chunk.text for chunk in chunks]

        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

        return cls(
            chunks=chunks,
            embeddings=np.asarray(embeddings, dtype=np.float32),
            model_name=model_name,
        )

    @classmethod
    def from_jsonl(
        cls,
        chunks_path: str | Path,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        batch_size: int = 32,
    ) -> "VectorRetriever":
        raw_chunks = read_jsonl(chunks_path)
        chunks = [DocumentChunk.model_validate(item) for item in raw_chunks]

        return cls.from_chunks(
            chunks=chunks,
            model_name=model_name,
            batch_size=batch_size,
        )

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
        )

        query_vector = np.asarray(query_embedding[0], dtype=np.float32)

        # Since vectors are normalized, dot product is cosine similarity.
        scores = self.embeddings @ query_vector

        ranked_indexes = np.argsort(scores)[::-1][:top_k]

        results: list[SearchResult] = []

        for rank, index in enumerate(ranked_indexes, start=1):
            score = float(scores[index])
            chunk = self.chunks[int(index)]

            results.append(
                SearchResult(
                    chunk_id=chunk.chunk_id,
                    score=score,
                    rank=rank,
                    method="vector",
                    chunk=chunk,
                )
            )

        return results

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("wb") as file:
            pickle.dump(
                {
                    "chunks": self.chunks,
                    "embeddings": self.embeddings,
                    "model_name": self.model_name,
                },
                file,
            )

    @classmethod
    def load(cls, path: str | Path) -> "VectorRetriever":
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Vector index not found: {path}")

        with path.open("rb") as file:
            payload = pickle.load(file)

        return cls(
            chunks=payload["chunks"],
            embeddings=payload["embeddings"],
            model_name=payload["model_name"],
        )
