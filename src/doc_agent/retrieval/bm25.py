from __future__ import annotations

import pickle
from pathlib import Path

from rank_bm25 import BM25Okapi

from doc_agent.models import DocumentChunk, SearchResult
from doc_agent.utils.jsonl import read_jsonl
from doc_agent.utils.text import tokenize_for_search


class BM25Retriever:
    def __init__(
        self,
        chunks: list[DocumentChunk],
        bm25: BM25Okapi,
    ) -> None:
        self.chunks = chunks
        self.bm25 = bm25

    @classmethod
    def from_chunks(cls, chunks: list[DocumentChunk]) -> "BM25Retriever":
        tokenized_corpus = [tokenize_for_search(chunk.text) for chunk in chunks]
        bm25 = BM25Okapi(tokenized_corpus)
        return cls(chunks=chunks, bm25=bm25)

    @classmethod
    def from_jsonl(cls, chunks_path: str | Path) -> "BM25Retriever":
        raw_chunks = read_jsonl(chunks_path)
        chunks = [DocumentChunk.model_validate(item) for item in raw_chunks]
        return cls.from_chunks(chunks)

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        tokenized_query = tokenize_for_search(query)
        scores = self.bm25.get_scores(tokenized_query)

        ranked_indexes = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )[:top_k]

        results: list[SearchResult] = []

        for rank, index in enumerate(ranked_indexes, start=1):
            score = float(scores[index])

            if score <= 0:
                continue

            chunk = self.chunks[index]

            results.append(
                SearchResult(
                    chunk_id=chunk.chunk_id,
                    score=score,
                    rank=rank,
                    method="bm25",
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
                    "bm25": self.bm25,
                },
                file,
            )

    @classmethod
    def load(cls, path: str | Path) -> "BM25Retriever":
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"BM25 index not found: {path}")

        with path.open("rb") as file:
            payload = pickle.load(file)

        return cls(
            chunks=payload["chunks"],
            bm25=payload["bm25"],
        )
