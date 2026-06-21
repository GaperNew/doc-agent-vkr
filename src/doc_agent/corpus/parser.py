from __future__ import annotations

import html
import re
from pathlib import Path

from bs4 import BeautifulSoup
from markdownify import markdownify as html_to_markdown

from doc_agent.models import DocumentPage


PERMALINK_RE = re.compile(r"\[¶\]\([^)]*\)")
IMAGE_MARKDOWN_RE = re.compile(r"!\[[^\]]*]\([^)]*\)")
HTML_TAG_RE = re.compile(r"</?[a-zA-Z][a-zA-Z0-9]*(?:\s[^>]*)?>")
MULTIPLE_BLANK_LINES_RE = re.compile(r"\n{3,}")


def _remove_noise(soup: BeautifulSoup) -> None:
    selectors = [
        "nav",
        "footer",
        "header",
        "script",
        "style",
        "noscript",
        ".md-sidebar",
        ".md-header",
        ".md-footer",
        ".md-tabs",
        ".md-top",
        ".md-search",
        ".md-skip",
        ".md-announce",
        ".md-source",
        ".md-content__button",
        ".md-dialog",
    ]

    for selector in selectors:
        for element in soup.select(selector):
            element.decompose()


def parse_html_file(
    html_path: str | Path,
    doc_id: str,
    source_name: str,
    source_url: str,
    title: str,
) -> DocumentPage:
    html_path = Path(html_path)

    if not html_path.exists():
        raise FileNotFoundError(f"HTML file not found: {html_path}")

    raw_html = html_path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(raw_html, "html.parser")

    _remove_noise(soup)

    main = soup.find("main")
    content = main if main is not None else soup.body or soup

    markdown_text = html_to_markdown(
        str(content),
        heading_style="ATX",
        bullets="-",
    )

    cleaned_text = clean_markdown_text(markdown_text)

    return DocumentPage(
        doc_id=doc_id,
        source_name=source_name,
        source_url=source_url,
        title=title,
        raw_html_path=str(html_path),
        text=cleaned_text,
        metadata={
            "parser": "beautifulsoup+markdownify",
        },
    )


def clean_markdown_text(text: str) -> str:
    text = html.unescape(text)

    # FastAPI docs add permalink symbols after headings: [¶](...)
    text = PERMALINK_RE.sub("", text)

    # Badges and images are mostly noise for technical retrieval.
    text = IMAGE_MARKDOWN_RE.sub("", text)

    # Remove remaining presentational HTML tags after markdownify.
    # This keeps Markdown autolinks like <http://127.0.0.1:8000/>.
    text = HTML_TAG_RE.sub("", text)

    lines: list[str] = []
    previous_blank = False

    for raw_line in text.splitlines():
        line = raw_line.rstrip()

        # Drop empty leftover Markdown image/link lines.
        if not line.strip():
            if not previous_blank:
                lines.append("")
            previous_blank = True
            continue

        # Drop lines that became only separators or badge remnants.
        if line.strip() in {"---", "----"}:
            lines.append(line.strip())
            previous_blank = False
            continue

        lines.append(line)
        previous_blank = False

    cleaned = "\n".join(lines).strip()
    cleaned = MULTIPLE_BLANK_LINES_RE.sub("\n\n", cleaned)

    return cleaned
