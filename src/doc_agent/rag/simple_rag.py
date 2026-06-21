from __future__ import annotations

import re

from doc_agent.llm.base import LLMClient
from doc_agent.models import AgentAnswer, SearchResult
from doc_agent.retrieval.bm25 import BM25Retriever
from doc_agent.retrieval.hybrid import HybridRetriever
from doc_agent.retrieval.vector import VectorRetriever
from doc_agent.rag.prompts import build_simple_rag_prompt
from doc_agent.utils.timing import measure_time


CJK_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


def contains_cjk(text: str) -> bool:
    return CJK_PATTERN.search(text) is not None


def remove_cjk_tail(text: str) -> str:
    """
    Qwen sometimes switches to Chinese in the middle of an otherwise valid answer.
    For the experiment we remove the contaminated tail instead of logging it.
    """
    match = CJK_PATTERN.search(text)

    if match is None:
        return text.strip()

    prefix = text[: match.start()].rstrip()

    # Remove the unfinished paragraph that immediately precedes the CJK text.
    paragraph_start = prefix.rfind("\n\n")

    if paragraph_start != -1:
        prefix = prefix[:paragraph_start].rstrip()

    return prefix.strip()


def infer_answer_status(answer_text: str) -> str:
    normalized = answer_text.strip().lower()

    insufficient_prefixes = [
        "по предоставленным фрагментам документации нельзя надежно ответить на вопрос",
        "по найденным фрагментам документации нельзя надежно ответить на вопрос",
        "по предоставленным фрагментам нельзя надежно ответить на вопрос",
    ]

    if any(normalized.startswith(prefix) for prefix in insufficient_prefixes):
        return "insufficient"

    return "answered"


def build_safe_insufficient_answer() -> str:
    return (
        "По предоставленным фрагментам документации нельзя надежно ответить на вопрос.\n"
        "В найденных фрагментах нет достаточных сведений для ответа на поставленный вопрос."
    )


class SimpleRAG:
    def __init__(
        self,
        retriever,
        llm_client: LLMClient,
        method_name: str,
        top_k: int = 5,
    ) -> None:
        self.retriever = retriever
        self.llm_client = llm_client
        self.method_name = method_name
        self.top_k = top_k

    @classmethod
    def with_bm25(
        cls,
        llm_client: LLMClient,
        top_k: int = 5,
    ) -> "SimpleRAG":
        return cls(
            retriever=BM25Retriever.load("data/indexes/bm25.pkl"),
            llm_client=llm_client,
            method_name="bm25_rag",
            top_k=top_k,
        )

    @classmethod
    def with_vector(
        cls,
        llm_client: LLMClient,
        top_k: int = 5,
    ) -> "SimpleRAG":
        return cls(
            retriever=VectorRetriever.load("data/indexes/vector.pkl"),
            llm_client=llm_client,
            method_name="vector_rag",
            top_k=top_k,
        )

    @classmethod
    def with_hybrid(
        cls,
        llm_client: LLMClient,
        top_k: int = 5,
    ) -> "SimpleRAG":
        return cls(
            retriever=HybridRetriever.load(),
            llm_client=llm_client,
            method_name="hybrid_rag",
            top_k=top_k,
        )

    def answer(self, question: str) -> AgentAnswer:
        with measure_time() as retrieval_timer:
            results: list[SearchResult] = self.retriever.search(
                question,
                top_k=self.top_k,
            )

        if not results:
            return AgentAnswer(
                question=question,
                method=self.method_name,
                status="insufficient",
                answer=build_safe_insufficient_answer(),
                sources=[],
                used_chunks=[],
                metadata={
                    "top_k": self.top_k,
                    "retrieved_chunks": 0,
                    "retrieval_elapsed_ms": retrieval_timer.elapsed_ms,
                    "llm_elapsed_ms": 0.0,
                    "total_elapsed_ms": retrieval_timer.elapsed_ms,
                    "prompt": None,
                    "raw_answer": None,
                    "answer_was_sanitized": False,
                    "retrieval_results": [],
                },
            )

        prompt = build_simple_rag_prompt(question, results)

        with measure_time() as llm_timer:
            raw_answer_text = self.llm_client.generate(prompt)

        total_elapsed_ms = round(
            retrieval_timer.elapsed_ms + llm_timer.elapsed_ms,
            3,
        )

        inferred_status = infer_answer_status(raw_answer_text)

        answer_was_sanitized = False

        if inferred_status == "insufficient":
            answer_text = build_safe_insufficient_answer()
            answer_was_sanitized = raw_answer_text.strip() != answer_text.strip()
        else:
            answer_text = raw_answer_text.strip()

            if contains_cjk(answer_text):
                answer_text = remove_cjk_tail(answer_text)
                answer_was_sanitized = True

        return AgentAnswer(
            question=question,
            method=self.method_name,
            status=inferred_status,
            answer=answer_text,
            sources=sorted({result.chunk.source_url for result in results}),
            used_chunks=[result.chunk_id for result in results],
            metadata={
                "top_k": self.top_k,
                "retrieved_chunks": len(results),
                "retrieval_elapsed_ms": retrieval_timer.elapsed_ms,
                "llm_elapsed_ms": llm_timer.elapsed_ms,
                "total_elapsed_ms": total_elapsed_ms,
                "prompt": prompt,
                "raw_answer": raw_answer_text,
                "answer_was_sanitized": answer_was_sanitized,
                "retrieval_results": [
                    {
                        "rank": result.rank,
                        "score": result.score,
                        "chunk_id": result.chunk_id,
                        "title": result.chunk.title,
                        "section_path": result.chunk.section_path,
                        "url": result.chunk.source_url,
                    }
                    for result in results
                ],
            },
        )