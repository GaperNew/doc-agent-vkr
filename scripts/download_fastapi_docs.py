from doc_agent.corpus.downloader import download_pages


def main() -> None:
    downloaded = download_pages(
        config_path="configs/sources_fastapi.json",
        output_dir="data/raw",
    )

    print(f"Downloaded pages: {len(downloaded)}")

    for page in downloaded:
        print(f"- {page['doc_id']} -> {page['raw_html_path']}")


if __name__ == "__main__":
    main()
