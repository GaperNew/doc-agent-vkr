from collections import Counter

from doc_agent.utils.jsonl import read_jsonl


def main() -> None:
    chunks = read_jsonl("data/processed/chunks.jsonl")

    print(f"Chunks count: {len(chunks)}")
    print()

    by_doc = Counter(chunk["doc_id"] for chunk in chunks)

    print("Chunks by document:")
    for doc_id, count in by_doc.most_common():
        print(f"- {doc_id}: {count}")

    print()
    print("=" * 100)
    print("First 5 chunks:")
    print("=" * 100)

    for chunk in chunks[:5]:
        print()
        print("-" * 100)
        print("chunk_id:", chunk["chunk_id"])
        print("doc_id:", chunk["doc_id"])
        print("title:", chunk["title"])
        print("section_path:", " / ".join(chunk["section_path"]))
        print("word_count:", chunk["metadata"].get("word_count"))
        print("url:", chunk["source_url"])
        print("-" * 100)
        print(chunk["text"][:1200])


if __name__ == "__main__":
    main()
