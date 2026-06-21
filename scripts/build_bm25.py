from doc_agent.retrieval.bm25 import BM25Retriever


def main() -> None:
    chunks_path = "data/processed/chunks.jsonl"
    index_path = "data/indexes/bm25.pkl"

    retriever = BM25Retriever.from_jsonl(chunks_path)
    retriever.save(index_path)

    print(f"BM25 index saved to: {index_path}")
    print(f"Chunks indexed: {len(retriever.chunks)}")


if __name__ == "__main__":
    main()
