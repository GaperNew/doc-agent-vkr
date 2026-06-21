from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import requests


DEFAULT_HEADERS = {
    "User-Agent": (
        "doc-agent-vkr/0.1 "
        "(research prototype; contact: local)"
    )
}


def load_sources_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Sources config not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def download_page(url: str, timeout: int = 30) -> str:
    response = requests.get(url, headers=DEFAULT_HEADERS, timeout=timeout)
    response.raise_for_status()
    return response.text


def download_pages(
    config_path: str | Path,
    output_dir: str | Path,
    sleep_seconds: float = 0.5,
) -> list[dict[str, Any]]:
    config = load_sources_config(config_path)

    source_name = config["source_name"]
    pages = config["pages"]

    output_dir = Path(output_dir)
    raw_dir = output_dir / source_name
    raw_dir.mkdir(parents=True, exist_ok=True)

    downloaded: list[dict[str, Any]] = []

    for page in pages:
        doc_id = page["doc_id"]
        url = page["url"]
        title = page.get("title", doc_id)

        print(f"Downloading: {url}")

        html = download_page(url)

        raw_html_path = raw_dir / f"{doc_id}.html"
        raw_html_path.write_text(html, encoding="utf-8")

        downloaded.append(
            {
                "doc_id": doc_id,
                "source_name": source_name,
                "source_url": url,
                "title": title,
                "raw_html_path": str(raw_html_path),
            }
        )

        time.sleep(sleep_seconds)

    return downloaded
