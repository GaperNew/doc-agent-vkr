from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from doc_agent.agent.agentic_rag import AgenticRAG
from doc_agent.llm.ollama_client import OllamaClient
from doc_agent.utils.jsonl import append_jsonl, read_jsonl


def utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_log_record(
    run_id: str,
    question_item: dict[str, Any],
    model: str,
    top_k: int,
    max_queries: int,
    result,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "created_at": utc_now_iso(),
        "question_id": question_item["question_id"],
        "question_type": question_item.get("question_type", "unknown"),
        "question": question_item["question"],
        "method": "agentic",
        "rag_method": result.method,
        "model": model,
        "top_k": top_k,
        "max_queries": max_queries,
        "status": result.status,
        "answer": result.answer,
        "sources": result.sources,
        "used_chunks": result.used_chunks,
        "expected_sources": question_item.get("expected_sources", []),
        "expected_answer_notes": question_item.get("expected_answer_notes", []),
        "metadata": result.metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--questions-path",
        type=str,
        default="experiments/questions/pilot_questions.jsonl",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-queries", type=int, default=3)
    parser.add_argument("--model", type=str, default="qwen2.5:7b-instruct")
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    questions = read_jsonl(args.questions_path)

    run_id = utc_now_compact()

    output_path = (
        args.output_path
        if args.output_path is not None
        else f"experiments/runs/run_agentic_{run_id}.jsonl"
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("Agentic experiment started")
    print("=" * 100)
    print(f"run_id: {run_id}")
    print(f"questions: {len(questions)}")
    print("method: agentic")
    print(f"model: {args.model}")
    print(f"top_k: {args.top_k}")
    print(f"max_queries: {args.max_queries}")
    print(f"output_path: {output_path}")
    print("=" * 100)

    llm = OllamaClient(model_name=args.model)

    print("Building AgenticRAG...", flush=True)
    rag = AgenticRAG.with_hybrid(
        llm_client=llm,
        top_k=args.top_k,
        max_queries=args.max_queries,
    )
    print("AgenticRAG built.", flush=True)

    total = len(questions)

    for index, question_item in enumerate(questions, start=1):
        question_id = question_item["question_id"]
        question = question_item["question"]

        print()
        print("-" * 100, flush=True)
        print(f"[{index}/{total}] agentic | {question_id}", flush=True)
        print(question, flush=True)

        result = rag.answer(question)

        log_record = make_log_record(
            run_id=run_id,
            question_item=question_item,
            model=args.model,
            top_k=args.top_k,
            max_queries=args.max_queries,
            result=result,
        )

        append_jsonl(output_path, log_record)

        print(f"status: {result.status}")
        print(f"sufficiency_reason: {result.metadata.get('sufficiency_reason')}")
        print(f"llm_attempts: {result.metadata.get('llm_attempts')}")
        print(f"answer_was_sanitized: {result.metadata.get('answer_was_sanitized')}")
        print(f"retrieval_elapsed_ms: {result.metadata.get('retrieval_elapsed_ms')}")
        print(f"llm_elapsed_ms: {result.metadata.get('llm_elapsed_ms')}")
        print(f"total_elapsed_ms: {result.metadata.get('total_elapsed_ms')}")
        print(f"queries: {result.metadata.get('queries')}")
        print(f"used_chunks: {', '.join(result.used_chunks)}")
        print("answer preview:")
        print(result.answer[:500].replace("\n", " "))

    print()
    print("=" * 100)
    print("Agentic experiment finished")
    print("=" * 100)
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
