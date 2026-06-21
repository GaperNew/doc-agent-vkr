from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from doc_agent.llm.ollama_client import OllamaClient
from doc_agent.rag.simple_rag import SimpleRAG
from doc_agent.utils.jsonl import append_jsonl, read_jsonl


def utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_rag(method: str, llm: OllamaClient, top_k: int) -> SimpleRAG:
    if method == "bm25":
        return SimpleRAG.with_bm25(llm_client=llm, top_k=top_k)

    if method == "vector":
        return SimpleRAG.with_vector(llm_client=llm, top_k=top_k)

    if method == "hybrid":
        return SimpleRAG.with_hybrid(llm_client=llm, top_k=top_k)

    raise ValueError(f"Unsupported method: {method}")


def normalize_methods(methods_raw: str) -> list[str]:
    methods = [method.strip() for method in methods_raw.split(",") if method.strip()]
    allowed = {"bm25", "vector", "hybrid"}

    invalid = [method for method in methods if method not in allowed]
    if invalid:
        raise ValueError(f"Invalid methods: {invalid}. Allowed: {sorted(allowed)}")

    return methods


def make_log_record(
    run_id: str,
    question_item: dict[str, Any],
    method: str,
    model: str,
    top_k: int,
    result,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "created_at": utc_now_iso(),
        "question_id": question_item["question_id"],
        "question_type": question_item.get("question_type", "unknown"),
        "question": question_item["question"],
        "method": method,
        "rag_method": result.method,
        "model": model,
        "top_k": top_k,
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
    parser.add_argument(
        "--methods",
        type=str,
        default="bm25,vector,hybrid",
        help="Comma-separated methods: bm25,vector,hybrid",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--model", type=str, default="qwen2.5:7b-instruct")
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    questions = read_jsonl(args.questions_path)
    methods = normalize_methods(args.methods)

    run_id = utc_now_compact()

    output_path = (
        args.output_path
        if args.output_path is not None
        else f"experiments/runs/run_{run_id}.jsonl"
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("Experiment started")
    print("=" * 100)
    print(f"run_id: {run_id}")
    print(f"questions: {len(questions)}")
    print(f"methods: {methods}")
    print(f"model: {args.model}")
    print(f"top_k: {args.top_k}")
    print(f"output_path: {output_path}")
    print("=" * 100)

    llm = OllamaClient(model_name=args.model)

    rag_by_method = {}

    for method in methods:
        print(f"Building RAG for method: {method}", flush=True)
        rag_by_method[method] = build_rag(method, llm=llm, top_k=args.top_k)
        print(f"RAG built for method: {method}", flush=True)

    total = len(questions) * len(methods)
    counter = 0

    for question_item in questions:
        question_id = question_item["question_id"]
        question = question_item["question"]

        for method in methods:
            counter += 1

            print()
            print("-" * 100, flush=True)
            print(f"[{counter}/{total}] {method} | {question_id}", flush=True)
            print(question, flush=True)

            rag = rag_by_method[method]
            result = rag.answer(question)

            log_record = make_log_record(
                run_id=run_id,
                question_item=question_item,
                method=method,
                model=args.model,
                top_k=args.top_k,
                result=result,
            )

            append_jsonl(output_path, log_record)

            print(f"status: {result.status}")
            print(f"retrieval_elapsed_ms: {result.metadata.get('retrieval_elapsed_ms')}")
            print(f"llm_elapsed_ms: {result.metadata.get('llm_elapsed_ms')}")
            print(f"total_elapsed_ms: {result.metadata.get('total_elapsed_ms')}")
            print(f"used_chunks: {', '.join(result.used_chunks)}")
            print("answer preview:")
            print(result.answer[:500].replace("\n", " "))

    print()
    print("=" * 100)
    print("Experiment finished")
    print("=" * 100)
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
