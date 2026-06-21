from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from doc_agent.agent.agentic_rag import AgenticRAG
from doc_agent.llm.ollama_client import OllamaClient
from doc_agent.utils.jsonl import append_jsonl


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def print_verified_sources(retrieval_results: list[dict]) -> None:
    print()
    print("=" * 100)
    print("Verified sources:")
    print("=" * 100)

    for item in retrieval_results:
        section_path = " / ".join(item["section_path"])

        print(
            f"[Source {item['rank']}] "
            f"{item['title']} | "
            f"{section_path} | "
            f"{item['chunk_id']} | "
            f"{item['url']}"
        )


def print_retrieval_results(retrieval_results: list[dict]) -> None:
    print()
    print("=" * 100)
    print("Retrieval results:")
    print("=" * 100)

    for item in retrieval_results:
        section_path = " / ".join(item["section_path"])

        print(
            f"{item['rank']}. "
            f"{item['chunk_id']} | "
            f"{item['title']} | "
            f"{section_path} | "
            f"score={item['score']:.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question", type=str)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-queries", type=int, default=3)
    parser.add_argument("--model", type=str, default="qwen2.5:7b-instruct")
    parser.add_argument(
        "--log-path",
        type=str,
        default="experiments/runs/ask_agentic_runs.jsonl",
    )

    args = parser.parse_args()

    llm = OllamaClient(model_name=args.model)

    rag = AgenticRAG.with_hybrid(
        llm_client=llm,
        top_k=args.top_k,
        max_queries=args.max_queries,
    )

    result = rag.answer(args.question)
    retrieval_results = result.metadata.get("retrieval_results", [])

    log_record = {
        "created_at": utc_now_iso(),
        "question": args.question,
        "method": "agentic",
        "rag_method": result.method,
        "model": args.model,
        "top_k": args.top_k,
        "max_queries": args.max_queries,
        "status": result.status,
        "answer": result.answer,
        "sources": result.sources,
        "used_chunks": result.used_chunks,
        "metadata": result.metadata,
    }

    append_jsonl(args.log_path, log_record)

    print("=" * 100)
    print(f"Method: {result.method}")
    print(f"Model: {args.model}")
    print(f"Status: {result.status}")
    print(f"Queries: {result.metadata.get('queries')}")
    print(f"Sufficiency reason: {result.metadata.get('sufficiency_reason')}")
    print(f"Used chunks: {', '.join(result.used_chunks)}")
    print(f"Retrieval time: {result.metadata.get('retrieval_elapsed_ms')} ms")
    print(f"LLM time: {result.metadata.get('llm_elapsed_ms')} ms")
    print(f"Total time: {result.metadata.get('total_elapsed_ms')} ms")
    print(f"Answer was sanitized: {result.metadata.get('answer_was_sanitized')}")
    print(f"Log path: {Path(args.log_path)}")
    print("=" * 100)
    print(result.answer)

    print_verified_sources(retrieval_results)
    print_retrieval_results(retrieval_results)


if __name__ == "__main__":
    main()
