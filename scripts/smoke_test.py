from doc_agent.models import DocumentChunk
from doc_agent.utils.jsonl import write_jsonl, read_jsonl


def main() -> None:
    chunk = DocumentChunk(
        chunk_id="test_chunk_001",
        doc_id="test_doc_001",
        source_name="test",
        source_url="https://example.com",
        title="Test document",
        section_path=["Test document"],
        text="This is a test chunk.",
        position=0,
    )

    path = "data/processed/smoke_test.jsonl"
    write_jsonl(path, [chunk.model_dump()])

    loaded = read_jsonl(path)

    print("Saved and loaded:", loaded[0]["chunk_id"])


if __name__ == "__main__":
    main()
