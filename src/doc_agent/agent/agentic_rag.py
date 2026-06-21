from __future__ import annotations

import re
from typing import Any

from doc_agent.llm.base import LLMClient
from doc_agent.models import AgentAnswer, SearchResult
from doc_agent.retrieval.hybrid import HybridRetriever
from doc_agent.rag.prompts import build_context
from doc_agent.rag.simple_rag import (
    build_safe_insufficient_answer,
    contains_cjk,
    infer_answer_status,
    remove_cjk_tail,
)
from doc_agent.utils.timing import measure_time


INSUFFICIENT_PHRASE = "По предоставленным фрагментам документации нельзя надежно ответить на вопрос."


def make_agent_step(
    role: str,
    action: str,
    status: str,
    output: Any,
) -> dict[str, Any]:
    return {
        "role": role,
        "action": action,
        "status": status,
        "output": output,
    }


def normalize_query(text: str) -> str:
    return " ".join(text.strip().split())


def deduplicate_results(results: list[SearchResult]) -> list[SearchResult]:
    seen_chunk_ids: set[str] = set()
    deduplicated: list[SearchResult] = []

    for result in results:
        if result.chunk_id in seen_chunk_ids:
            continue

        seen_chunk_ids.add(result.chunk_id)
        deduplicated.append(result)

    return deduplicated


def infer_question_keywords(question: str) -> list[str]:
    technical_tokens = re.findall(
        r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*",
        question,
    )

    stop_words = {
        "how",
        "what",
        "why",
        "when",
        "where",
        "which",
        "to",
        "in",
        "for",
        "with",
        "and",
        "or",
        "the",
        "a",
        "an",
        "is",
        "are",
        "of",
        "this",
        "that",
        "according",
        "documentation",
        "happens",
        "user",
        "sends",
        "if",
        "as",
    }

    keywords: list[str] = []

    for token in technical_tokens:
        lowered = token.lower()

        if lowered in stop_words:
            continue

        if token not in keywords:
            keywords.append(token)

    return keywords


def build_follow_up_queries(question: str) -> list[str]:
    question = normalize_query(question)
    keywords = infer_question_keywords(question)

    queries = [question]

    if keywords:
        queries.append(" ".join(keywords))

    if len(keywords) >= 2:
        queries.append(" ".join(keywords[:2]))

    return list(dict.fromkeys(queries))


def is_obviously_out_of_scope(question: str, results: list[SearchResult]) -> bool:
    question_lower = question.lower()

    hard_out_of_scope_terms = [
        "django",
        "sqlalchemy",
        "postgresql",
        "postgres",
        "orm",
        "kubernetes",
    ]

    if not any(term in question_lower for term in hard_out_of_scope_terms):
        return False

    combined_context = " ".join(
        [
            result.chunk.title.lower()
            + " "
            + " ".join(result.chunk.section_path).lower()
            + " "
            + result.chunk.text.lower()
            for result in results
        ]
    )

    for term in hard_out_of_scope_terms:
        if term in question_lower and term not in combined_context:
            return True

    return False


def has_keyword_support(question: str, results: list[SearchResult]) -> bool:
    keywords = infer_question_keywords(question)

    if not keywords:
        return True

    combined_context = " ".join(
        [
            result.chunk.title
            + " "
            + " ".join(result.chunk.section_path)
            + " "
            + result.chunk.text
            for result in results
        ]
    ).lower()

    supported_keywords = 0

    for keyword in keywords:
        if keyword.lower() in combined_context:
            supported_keywords += 1

    return supported_keywords >= max(1, min(2, len(keywords)))


def remove_contradictory_insufficient_tail(text: str) -> str:
    """
    Sometimes the model gives a useful answer and then appends a contradictory
    insufficient-information sentence. For the experiment, remove that tail.
    """
    text = text.strip()

    if text.startswith(INSUFFICIENT_PHRASE):
        return text

    marker_variants = [
        "\n\n" + INSUFFICIENT_PHRASE,
        "\n" + INSUFFICIENT_PHRASE,
        INSUFFICIENT_PHRASE,
    ]

    cut_positions = [
        text.find(marker)
        for marker in marker_variants
        if text.find(marker) > 0
    ]

    if not cut_positions:
        return text

    cut_at = min(cut_positions)
    return text[:cut_at].rstrip()


def build_agentic_answer_prompt(
    question: str,
    results: list[SearchResult],
    queries: list[str],
    is_retry: bool = False,
) -> str:
    context = build_context(results)

    retry_note = ""

    if is_retry:
        retry_note = """
В предыдущей попытке ты ошибочно начал ответ с отказа, хотя найденные фрагменты содержат нужные сведения.
Сейчас нужно дать технический ответ по найденным фрагментам.
Не начинай ответ с фразы о недостаточности информации.
""".strip()

    return f"""
Ты — технический ассистент по документации FastAPI.

Перед генерацией уже выполнена агентная проверка достаточности:
- выполнены несколько поисковых запросов;
- найдены тематически релевантные фрагменты;
- ключевые термины вопроса подтверждены найденным контекстом;
- вопрос не признан выходящим за пределы найденной документации.

{retry_note}

Твоя задача — ответить на вопрос пользователя, используя ТОЛЬКО предоставленные фрагменты.

Правила:
- Отвечай только на русском языке.
- Не используй внешние знания.
- Не добавляй технологии, библиотеки, параметры или команды, которых нет во фрагментах.
- Не пиши URL.
- Не создавай отдельный список источников.
- Подкрепляй ключевые утверждения маркерами [Source N].
- Не добавляй китайский, английский или другой язык, кроме кода, имён классов, параметров и команд.
- Если во фрагментах есть пример кода, используй его.
- Если во фрагментах есть только часть ответа, ответь только по этой части.
- Не пиши фразу "{INSUFFICIENT_PHRASE}", если во фрагментах есть класс, параметр, пример кода или объяснение, позволяющее ответить.

Поисковые запросы агента:
{queries}

Вопрос пользователя:
{question}

Фрагменты документации:
{context}

Ответ:
""".strip()


class AgenticRAG:
    def __init__(
        self,
        retriever: HybridRetriever,
        llm_client: LLMClient,
        top_k: int = 5,
        max_queries: int = 3,
    ) -> None:
        self.retriever = retriever
        self.llm_client = llm_client
        self.top_k = top_k
        self.max_queries = max_queries
        self.method_name = "agentic_rag"

    @classmethod
    def with_hybrid(
        cls,
        llm_client: LLMClient,
        top_k: int = 5,
        max_queries: int = 3,
    ) -> "AgenticRAG":
        return cls(
            retriever=HybridRetriever.load(),
            llm_client=llm_client,
            top_k=top_k,
            max_queries=max_queries,
        )

    def retrieve_multi_step(self, question: str) -> tuple[list[SearchResult], list[str]]:
        queries = build_follow_up_queries(question)[: self.max_queries]

        all_results: list[SearchResult] = []

        for query in queries:
            query_results = self.retriever.search(
                query,
                top_k=self.top_k,
            )
            all_results.extend(query_results)

        deduplicated = deduplicate_results(all_results)

        reranked: list[SearchResult] = []

        for rank, result in enumerate(deduplicated[: self.top_k], start=1):
            reranked.append(
                SearchResult(
                    chunk_id=result.chunk_id,
                    score=result.score,
                    rank=rank,
                    method=self.method_name,
                    chunk=result.chunk,
                )
            )

        return reranked, queries

    def answer(self, question: str) -> AgentAnswer:
        agent_steps: list[dict[str, Any]] = []

        with measure_time() as retrieval_timer:
            results, queries = self.retrieve_multi_step(question)

        agent_steps.append(
            make_agent_step(
                role="query_planner",
                action="build_follow_up_queries",
                status="completed",
                output={
                    "queries": queries,
                    "max_queries": self.max_queries,
                },
            )
        )

        serialized_results = self._serialize_results(results)

        agent_steps.append(
            make_agent_step(
                role="retrieval_specialist",
                action="multi_step_hybrid_retrieval",
                status="completed",
                output={
                    "retrieved_chunks": len(results),
                    "selected_chunk_ids": [result.chunk_id for result in results],
                    "top_k": self.top_k,
                },
            )
        )

        if not results:
            agent_steps.append(
                make_agent_step(
                    role="scope_checker",
                    action="out_of_scope_check",
                    status="skipped",
                    output="no retrieved chunks",
                )
            )
            agent_steps.append(
                make_agent_step(
                    role="sufficiency_checker",
                    action="keyword_support_check",
                    status="failed",
                    output="no retrieved chunks",
                )
            )
            agent_steps.append(
                make_agent_step(
                    role="answer_generator",
                    action="llm_generation",
                    status="skipped",
                    output="insufficient context",
                )
            )
            agent_steps.append(
                make_agent_step(
                    role="verifier",
                    action="final_answer_validation",
                    status="completed",
                    output="safe insufficient answer returned",
                )
            )

            return AgentAnswer(
                question=question,
                method=self.method_name,
                status="insufficient",
                answer=build_safe_insufficient_answer(),
                sources=[],
                used_chunks=[],
                metadata={
                    "top_k": self.top_k,
                    "max_queries": self.max_queries,
                    "queries": queries,
                    "agent_steps": agent_steps,
                    "retrieved_chunks": 0,
                    "retrieval_elapsed_ms": retrieval_timer.elapsed_ms,
                    "llm_elapsed_ms": 0.0,
                    "total_elapsed_ms": retrieval_timer.elapsed_ms,
                    "prompt": None,
                    "raw_answer": None,
                    "retry_raw_answer": None,
                    "llm_attempts": 0,
                    "answer_was_sanitized": False,
                    "sufficiency_reason": "no_results",
                    "retrieval_results": [],
                },
            )

        is_out_of_scope = is_obviously_out_of_scope(question, results)

        agent_steps.append(
            make_agent_step(
                role="scope_checker",
                action="out_of_scope_check",
                status="failed" if is_out_of_scope else "passed",
                output={
                    "is_out_of_scope": is_out_of_scope,
                    "reason": (
                        "question contains external technology not found in context"
                        if is_out_of_scope
                        else "question is within retrieved documentation scope"
                    ),
                },
            )
        )

        if is_out_of_scope:
            agent_steps.append(
                make_agent_step(
                    role="sufficiency_checker",
                    action="keyword_support_check",
                    status="skipped",
                    output="scope check failed",
                )
            )
            agent_steps.append(
                make_agent_step(
                    role="answer_generator",
                    action="llm_generation",
                    status="skipped",
                    output="out-of-scope question",
                )
            )
            agent_steps.append(
                make_agent_step(
                    role="verifier",
                    action="final_answer_validation",
                    status="completed",
                    output="safe insufficient answer returned without LLM call",
                )
            )

            return AgentAnswer(
                question=question,
                method=self.method_name,
                status="insufficient",
                answer=build_safe_insufficient_answer(),
                sources=sorted({result.chunk.source_url for result in results}),
                used_chunks=[result.chunk_id for result in results],
                metadata={
                    "top_k": self.top_k,
                    "max_queries": self.max_queries,
                    "queries": queries,
                    "agent_steps": agent_steps,
                    "retrieved_chunks": len(results),
                    "retrieval_elapsed_ms": retrieval_timer.elapsed_ms,
                    "llm_elapsed_ms": 0.0,
                    "total_elapsed_ms": retrieval_timer.elapsed_ms,
                    "prompt": None,
                    "raw_answer": None,
                    "retry_raw_answer": None,
                    "llm_attempts": 0,
                    "answer_was_sanitized": True,
                    "sufficiency_reason": "out_of_scope_keyword",
                    "retrieval_results": serialized_results,
                },
            )

        keyword_support = has_keyword_support(question, results)

        agent_steps.append(
            make_agent_step(
                role="sufficiency_checker",
                action="keyword_support_check",
                status="passed" if keyword_support else "failed",
                output={
                    "has_keyword_support": keyword_support,
                    "keywords": infer_question_keywords(question),
                },
            )
        )

        if not keyword_support:
            agent_steps.append(
                make_agent_step(
                    role="answer_generator",
                    action="llm_generation",
                    status="skipped",
                    output="weak keyword support",
                )
            )
            agent_steps.append(
                make_agent_step(
                    role="verifier",
                    action="final_answer_validation",
                    status="completed",
                    output="safe insufficient answer returned without LLM call",
                )
            )

            return AgentAnswer(
                question=question,
                method=self.method_name,
                status="insufficient",
                answer=build_safe_insufficient_answer(),
                sources=sorted({result.chunk.source_url for result in results}),
                used_chunks=[result.chunk_id for result in results],
                metadata={
                    "top_k": self.top_k,
                    "max_queries": self.max_queries,
                    "queries": queries,
                    "agent_steps": agent_steps,
                    "retrieved_chunks": len(results),
                    "retrieval_elapsed_ms": retrieval_timer.elapsed_ms,
                    "llm_elapsed_ms": 0.0,
                    "total_elapsed_ms": retrieval_timer.elapsed_ms,
                    "prompt": None,
                    "raw_answer": None,
                    "retry_raw_answer": None,
                    "llm_attempts": 0,
                    "answer_was_sanitized": True,
                    "sufficiency_reason": "weak_keyword_support",
                    "retrieval_results": serialized_results,
                },
            )

        prompt = build_agentic_answer_prompt(
            question=question,
            results=results,
            queries=queries,
            is_retry=False,
        )

        with measure_time() as first_llm_timer:
            raw_answer_text = self.llm_client.generate(prompt)

        first_answer_text = raw_answer_text.strip()

        answer_was_sanitized = False
        verifier_actions: list[str] = []

        if contains_cjk(first_answer_text):
            first_answer_text = remove_cjk_tail(first_answer_text)
            answer_was_sanitized = True
            verifier_actions.append("removed_cjk_tail")

        cleaned_first_answer_text = remove_contradictory_insufficient_tail(first_answer_text)

        if cleaned_first_answer_text != first_answer_text:
            answer_was_sanitized = True
            verifier_actions.append("removed_contradictory_insufficient_tail")

        first_answer_text = cleaned_first_answer_text
        first_status = infer_answer_status(first_answer_text)

        retry_raw_answer_text = None
        retry_prompt = None
        retry_elapsed_ms = 0.0
        llm_attempts = 1

        agent_steps.append(
            make_agent_step(
                role="answer_generator",
                action="llm_generation",
                status="completed",
                output={
                    "attempt": 1,
                    "inferred_status": first_status,
                    "elapsed_ms": first_llm_timer.elapsed_ms,
                },
            )
        )

        if first_status == "insufficient":
            retry_prompt = build_agentic_answer_prompt(
                question=question,
                results=results,
                queries=queries,
                is_retry=True,
            )

            with measure_time() as retry_llm_timer:
                retry_raw_answer_text = self.llm_client.generate(retry_prompt)

            retry_elapsed_ms = retry_llm_timer.elapsed_ms
            llm_attempts = 2

            retry_answer_text = retry_raw_answer_text.strip()

            if contains_cjk(retry_answer_text):
                retry_answer_text = remove_cjk_tail(retry_answer_text)
                answer_was_sanitized = True
                verifier_actions.append("removed_cjk_tail_after_retry")

            cleaned_retry_answer_text = remove_contradictory_insufficient_tail(
                retry_answer_text
            )

            if cleaned_retry_answer_text != retry_answer_text:
                answer_was_sanitized = True
                verifier_actions.append("removed_contradictory_tail_after_retry")

            retry_answer_text = cleaned_retry_answer_text
            retry_status = infer_answer_status(retry_answer_text)

            agent_steps.append(
                make_agent_step(
                    role="answer_generator",
                    action="llm_retry_generation",
                    status="completed",
                    output={
                        "attempt": 2,
                        "inferred_status": retry_status,
                        "elapsed_ms": retry_elapsed_ms,
                    },
                )
            )

            if retry_status == "insufficient":
                final_status = "insufficient"
                answer_text = build_safe_insufficient_answer()
                answer_was_sanitized = True
                sufficiency_reason = "model_refused_after_retry"
                verifier_actions.append("safe_insufficient_after_retry")
            else:
                final_status = "answered"
                answer_text = retry_answer_text
                sufficiency_reason = "generated_answer_after_retry"
        else:
            final_status = "answered"
            answer_text = first_answer_text
            sufficiency_reason = "generated_answer"

        if not verifier_actions:
            verifier_actions.append("no_postprocessing_required")

        agent_steps.append(
            make_agent_step(
                role="verifier",
                action="final_answer_validation",
                status="completed",
                output={
                    "final_status": final_status,
                    "answer_was_sanitized": answer_was_sanitized,
                    "verifier_actions": verifier_actions,
                    "sufficiency_reason": sufficiency_reason,
                },
            )
        )

        llm_elapsed_ms = round(first_llm_timer.elapsed_ms + retry_elapsed_ms, 3)
        total_elapsed_ms = round(retrieval_timer.elapsed_ms + llm_elapsed_ms, 3)

        return AgentAnswer(
            question=question,
            method=self.method_name,
            status=final_status,
            answer=answer_text,
            sources=sorted({result.chunk.source_url for result in results}),
            used_chunks=[result.chunk_id for result in results],
            metadata={
                "top_k": self.top_k,
                "max_queries": self.max_queries,
                "queries": queries,
                "agent_steps": agent_steps,
                "retrieved_chunks": len(results),
                "retrieval_elapsed_ms": retrieval_timer.elapsed_ms,
                "llm_elapsed_ms": llm_elapsed_ms,
                "total_elapsed_ms": total_elapsed_ms,
                "prompt": prompt,
                "retry_prompt": retry_prompt,
                "raw_answer": raw_answer_text,
                "retry_raw_answer": retry_raw_answer_text,
                "llm_attempts": llm_attempts,
                "answer_was_sanitized": answer_was_sanitized,
                "sufficiency_reason": sufficiency_reason,
                "retrieval_results": serialized_results,
            },
        )

    def _serialize_results(self, results: list[SearchResult]) -> list[dict[str, Any]]:
        return [
            {
                "rank": result.rank,
                "score": result.score,
                "chunk_id": result.chunk_id,
                "title": result.chunk.title,
                "section_path": result.chunk.section_path,
                "url": result.chunk.source_url,
            }
            for result in results
        ]