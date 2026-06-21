from __future__ import annotations

from doc_agent.retrieval.bm25 import BM25Retriever
from doc_agent.retrieval.hybrid import HybridRetriever
from doc_agent.retrieval.vector import VectorRetriever


QUERIES = [
    "CORSMiddleware allow_origins",
    "how to allow browser requests from frontend running on another domain",
    "path parameter item_id int validation",
    "Docker image FastAPI deployment",
    "dependency injection Depends",
    "OAuth2PasswordBearer tokenUrl",
]


def print_compact_results(method_name: str, results) -> None:
    print(f"\n{method_name}")
    print("-" * 100)

    for result in results[:3]:
        chunk = result.chunk

        print(
            f"{result.rank}. "
            f"{chunk.chunk_id} | "
            f"{chunk.title} | "
            f"{' / '.join(chunk.section_path)} | "
            f"score={result.score:.4f}"
        )


def main() -> None:
    bm25 = BM25Retriever.load("data/indexes/bm25.pkl")
    vector = VectorRetriever.load("data/indexes/vector.pkl")
    hybrid = HybridRetriever.load()

    for query in QUERIES:
        print("=" * 120)
        print(f"QUERY: {query}")
        print("=" * 120)

        print_compact_results("BM25", bm25.search(query, top_k=3))
        print_compact_results("VECTOR", vector.search(query, top_k=3))
        print_compact_results("HYBRID", hybrid.search(query, top_k=3))


if __name__ == "__main__":
    main()
