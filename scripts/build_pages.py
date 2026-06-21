from pathlib import Path

from doc_agent.corpus.downloader import load_sources_config
from doc_agent.corpus.parser import parse_html_file
from doc_agent.utils.jsonl import write_jsonl


def main() -> None:
    config = load_sources_config("configs/sources_fastapi.json")

    source_name = config["source_name"]
    pages_config = config["pages"]

    parsed_pages = []

    for page in pages_config:
        doc_id = page["doc_id"]
        title = page["title"]
        url = page["url"]

        html_path = Path("data/raw") / source_name / f"{doc_id}.html"

        parsed = parse_html_file(
            html_path=html_path,
            doc_id=doc_id,
            source_name=source_name,
            source_url=url,
            title=title,
        )

        parsed_pages.append(parsed.model_dump())

        print(
            f"Parsed {doc_id}: "
            f"{len(parsed.text)} chars, "
            f"{len(parsed.text.split())} words"
        )

    output_path = "data/processed/pages.jsonl"
    write_jsonl(output_path, parsed_pages)

    print(f"Saved pages to: {output_path}")


if __name__ == "__main__":
    main()
