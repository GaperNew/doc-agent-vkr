from __future__ import annotations

import argparse

from doc_agent.retrieval.hybrid import HybridRetriever


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query", type=str)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    retriever = HybridRetriever.load()
    results = retriever.search(args.query, top_k=args.top_k)

    print(f"Query: {args.query}")
    print(f"Results: {len(results)}")
    print()

    for result in results:
        chunk = result.chunk

        print("=" * 100)
        print(f"Rank: {result.rank}")
        print(f"Score: {result.score:.6f}")
        print(f"Chunk ID: {chunk.chunk_id}")
        print(f"Title: {chunk.title}")
        print(f"Section: {' / '.join(chunk.section_path)}")
        print(f"URL: {chunk.source_url}")
        print("-" * 100)
        print(chunk.text[:1500])
        print()


if __name__ == "__main__":
    main()
