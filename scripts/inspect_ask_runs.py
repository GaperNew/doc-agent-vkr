from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    path = Path("experiments/runs/ask_runs.jsonl")

    if not path.exists():
        print(f"Log file not found: {path}")
        return

    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    if not lines:
        print("Log file is empty.")
        return

    record = json.loads(lines[-1])

    print("=" * 100)
    print("Last ask.py run")
    print("=" * 100)
    print("created_at:", record["created_at"])
    print("method:", record["rag_method"])
    print("model:", record["model"])
    print("status:", record["status"])
    print("question:", record["question"])
    print("retrieval_elapsed_ms:", record["metadata"].get("retrieval_elapsed_ms"))
    print("llm_elapsed_ms:", record["metadata"].get("llm_elapsed_ms"))
    print("total_elapsed_ms:", record["metadata"].get("total_elapsed_ms"))

    print()
    print("=" * 100)
    print("Answer")
    print("=" * 100)
    print(record["answer"])

    print()
    print("=" * 100)
    print("Retrieval results")
    print("=" * 100)

    for item in record["metadata"].get("retrieval_results", []):
        print(
            f"{item['rank']}. {item['chunk_id']} | "
            f"{item['title']} | "
            f"{' / '.join(item['section_path'])} | "
            f"{item['url']} | "
            f"score={item['score']:.4f}"
        )


if __name__ == "__main__":
    main()
