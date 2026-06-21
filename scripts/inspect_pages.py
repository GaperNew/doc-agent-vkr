from doc_agent.utils.jsonl import read_jsonl


def main() -> None:
    pages = read_jsonl("data/processed/pages.jsonl")

    print(f"Pages count: {len(pages)}")
    print()

    for page in pages[:3]:
        print("=" * 80)
        print(page["doc_id"])
        print(page["title"])
        print(page["source_url"])
        print("-" * 80)
        print(page["text"][:1500])


if __name__ == "__main__":
    main()
