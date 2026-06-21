from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


QuestionType = Literal[
    "fact",
    "instructional",
    "diagnostic",
    "comparative",
    "multi_step",
    "insufficient_info",
    "unknown",
]


AnswerStatus = Literal[
    "answered",
    "partially_answered",
    "insufficient",
    "contradiction",
    "out_of_scope",
    "error",
]


class DocumentPage(BaseModel):
    doc_id: str
    source_name: str
    source_url: str
    title: str
    raw_html_path: str | None = None
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentChunk(BaseModel):
    chunk_id: str
    doc_id: str
    source_name: str
    source_url: str
    title: str
    section_path: list[str] = Field(default_factory=list)
    text: str
    position: int
    content_type: str = "text"
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchResult(BaseModel):
    chunk_id: str
    score: float
    rank: int
    method: str
    chunk: DocumentChunk


class AgentAnswer(BaseModel):
    question: str
    method: str
    status: AnswerStatus
    answer: str
    sources: list[str] = Field(default_factory=list)
    used_chunks: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
