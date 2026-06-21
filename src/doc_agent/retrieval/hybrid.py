from __future__ import annotations

from dataclasses import dataclass

from doc_agent.models import SearchResult
from doc_agent.retrieval.bm25 import BM25Retriever
from doc_agent.retrieval.vector import VectorRetriever


@dataclass
class HybridSearchConfig:
    bm25_top_k: int = 10
    vector_top_k: int = 10
    final_top_k: int = 5
    rrf_k: int = 60


class HybridRetriever:
    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        vector_retriever: VectorRetriever,
        config: HybridSearchConfig | None = None,
    ) -> None:
        self.bm25_retriever = bm25_retriever
        self.vector_retriever = vector_retriever
        self.config = config or HybridSearchConfig()

    @classmethod
    def load(
        cls,
        bm25_path: str = "data/indexes/bm25.pkl",
        vector_path: str = "data/indexes/vector.pkl",
        config: HybridSearchConfig | None = None,
    ) -> "HybridRetriever":
        return cls(
            bm25_retriever=BM25Retriever.load(bm25_path),
            vector_retriever=VectorRetriever.load(vector_path),
            config=config,
        )

    def search(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        final_top_k = top_k or self.config.final_top_k

        bm25_results = self.bm25_retriever.search(
            query,
            top_k=self.config.bm25_top_k,
        )
        vector_results = self.vector_retriever.search(
            query,
            top_k=self.config.vector_top_k,
        )

        # Reciprocal Rank Fusion:
        # score = sum(1 / (k + rank_i))
        fused_scores: dict[str, float] = {}
        best_result_by_chunk_id: dict[str, SearchResult] = {}

        for result in bm25_results:
            fused_scores[result.chunk_id] = fused_scores.get(result.chunk_id, 0.0) + (
                1.0 / (self.config.rrf_k + result.rank)
            )
            best_result_by_chunk_id[result.chunk_id] = result

        for result in vector_results:
            fused_scores[result.chunk_id] = fused_scores.get(result.chunk_id, 0.0) + (
                1.0 / (self.config.rrf_k + result.rank)
            )

            if result.chunk_id not in best_result_by_chunk_id:
                best_result_by_chunk_id[result.chunk_id] = result

        ranked_chunk_ids = sorted(
            fused_scores,
            key=lambda chunk_id: fused_scores[chunk_id],
            reverse=True,
        )[:final_top_k]

        hybrid_results: list[SearchResult] = []

        for rank, chunk_id in enumerate(ranked_chunk_ids, start=1):
            base_result = best_result_by_chunk_id[chunk_id]

            hybrid_results.append(
                SearchResult(
                    chunk_id=chunk_id,
                    score=fused_scores[chunk_id],
                    rank=rank,
                    method="hybrid_rrf",
                    chunk=base_result.chunk,
                )
            )

        return hybrid_results
