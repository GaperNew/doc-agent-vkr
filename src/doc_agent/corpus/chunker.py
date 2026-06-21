from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from doc_agent.models import DocumentChunk, DocumentPage


HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]*\)")


@dataclass
class MarkdownSection:
    section_path: list[str]
    text: str


def clean_heading(heading: str) -> str:
    heading = MARKDOWN_LINK_RE.sub(r"\1", heading)
    heading = heading.replace("#", "").strip()
    return heading


def split_markdown_sections(text: str, fallback_title: str) -> list[MarkdownSection]:
    sections: list[MarkdownSection] = []
    heading_stack: list[str] = []
    current_lines: list[str] = []

    def flush_current() -> None:
        nonlocal current_lines

        section_text = "\n".join(current_lines).strip()
        if section_text:
            path = heading_stack.copy() or [fallback_title]
            sections.append(MarkdownSection(section_path=path, text=section_text))

        current_lines = []

    for line in text.splitlines():
        match = HEADING_RE.match(line)

        if match:
            flush_current()

            level = len(match.group(1))
            heading = clean_heading(match.group(2))

            heading_stack[:] = heading_stack[: level - 1]
            heading_stack.append(heading)

            current_lines.append(line)
            continue

        current_lines.append(line)

    flush_current()

    if not sections and text.strip():
        sections.append(
            MarkdownSection(
                section_path=[fallback_title],
                text=text.strip(),
            )
        )

    return sections


def count_words(text: str) -> int:
    return len(text.split())


def split_long_text_by_paragraphs(
    text: str,
    max_words: int,
    overlap_words: int,
) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks: list[str] = []
    current_parts: list[str] = []
    current_words = 0

    for paragraph in paragraphs:
        paragraph_words = count_words(paragraph)

        if paragraph_words > max_words:
            if current_parts:
                chunks.append("\n\n".join(current_parts).strip())
                current_parts = []
                current_words = 0

            chunks.extend(split_long_paragraph(paragraph, max_words, overlap_words))
            continue

        if current_parts and current_words + paragraph_words > max_words:
            chunks.append("\n\n".join(current_parts).strip())

            overlap_text = tail_words("\n\n".join(current_parts), overlap_words)
            current_parts = [overlap_text] if overlap_text else []
            current_words = count_words(overlap_text)

        current_parts.append(paragraph)
        current_words += paragraph_words

    if current_parts:
        chunks.append("\n\n".join(current_parts).strip())

    return [chunk for chunk in chunks if chunk.strip()]


def split_long_paragraph(
    paragraph: str,
    max_words: int,
    overlap_words: int,
) -> list[str]:
    words = paragraph.split()
    chunks: list[str] = []

    step = max(max_words - overlap_words, 1)

    for start in range(0, len(words), step):
        end = start + max_words
        part = " ".join(words[start:end]).strip()

        if part:
            chunks.append(part)

        if end >= len(words):
            break

    return chunks


def tail_words(text: str, words_count: int) -> str:
    if words_count <= 0:
        return ""

    words = text.split()
    return " ".join(words[-words_count:])


def make_text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def chunk_document_page(
    page: DocumentPage,
    max_words: int = 260,
    overlap_words: int = 40,
) -> list[DocumentChunk]:
    sections = split_markdown_sections(page.text, fallback_title=page.title)

    chunks: list[DocumentChunk] = []
    position = 0

    for section in sections:
        section_texts = split_long_text_by_paragraphs(
            section.text,
            max_words=max_words,
            overlap_words=overlap_words,
        )

        for text_part in section_texts:
            section_prefix = " / ".join(section.section_path)
            chunk_text = f"Section: {section_prefix}\n\n{text_part}".strip()

            chunk = DocumentChunk(
                chunk_id=f"{page.doc_id}_chunk_{position:04d}",
                doc_id=page.doc_id,
                source_name=page.source_name,
                source_url=page.source_url,
                title=page.title,
                section_path=section.section_path,
                text=chunk_text,
                position=position,
                content_type="text",
                metadata={
                    **page.metadata,
                    "text_hash": make_text_hash(chunk_text),
                    "word_count": count_words(chunk_text),
                    "max_words": max_words,
                    "overlap_words": overlap_words,
                },
            )

            chunks.append(chunk)
            position += 1

    return chunks
