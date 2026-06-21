from doc_agent.corpus.chunker import chunk_document_page
from doc_agent.models import DocumentPage
from doc_agent.utils.jsonl import read_jsonl, write_jsonl


def main() -> None:
    pages_raw = read_jsonl("data/processed/pages.jsonl")

    all_chunks = []

    for page_raw in pages_raw:
        page = DocumentPage.model_validate(page_raw)
        chunks = chunk_document_page(page)

        all_chunks.extend(chunk.model_dump() for chunk in chunks)

        print(f"{page.doc_id}: {len(chunks)} chunks")

    output_path = "data/processed/chunks.jsonl"
    write_jsonl(output_path, all_chunks)

    print()
    print(f"Total chunks: {len(all_chunks)}")
    print(f"Saved chunks to: {output_path}")


if __name__ == "__main__":
    main()
