from doc_agent.retrieval.vector import VectorRetriever


def main() -> None:
    chunks_path = "data/processed/chunks.jsonl"
    index_path = "data/indexes/vector.pkl"

    retriever = VectorRetriever.from_jsonl(chunks_path)
    retriever.save(index_path)

    print(f"Vector index saved to: {index_path}")
    print(f"Chunks indexed: {len(retriever.chunks)}")
    print(f"Embedding model: {retriever.model_name}")
    print(f"Embeddings shape: {retriever.embeddings.shape}")


if __name__ == "__main__":
    main()
