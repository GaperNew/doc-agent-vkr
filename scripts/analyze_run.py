from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            records.append(json.loads(line))

    return records


def expected_source_hit(record: dict[str, Any]) -> bool | None:
    expected_sources = record.get("expected_sources", [])
    actual_sources = record.get("sources", [])

    if not expected_sources:
        return None

    return bool(set(expected_sources) & set(actual_sources))


def get_retrieved_titles(record: dict[str, Any]) -> str:
    retrieval_results = record.get("metadata", {}).get("retrieval_results", [])

    titles = []
    for item in retrieval_results:
        title = item.get("title", "")
        chunk_id = item.get("chunk_id", "")
        if title and chunk_id:
            titles.append(f"{title} ({chunk_id})")

    return " | ".join(titles)


def get_top_chunk(record: dict[str, Any]) -> str:
    retrieval_results = record.get("metadata", {}).get("retrieval_results", [])

    if not retrieval_results:
        return ""

    top = retrieval_results[0]
    return f"{top.get('title', '')} / {top.get('chunk_id', '')}"


def make_row(record: dict[str, Any]) -> dict[str, Any]:
    metadata = record.get("metadata", {})

    answer = record.get("answer", "")
    short_answer = answer.replace("\n", " ").strip()

    if len(short_answer) > 300:
        short_answer = short_answer[:300] + "..."

    return {
        "run_id": record.get("run_id"),
        "question_id": record.get("question_id"),
        "question_type": record.get("question_type"),
        "method": record.get("method"),
        "rag_method": record.get("rag_method"),
        "status": record.get("status"),
        "expected_source_hit": expected_source_hit(record),
        "answer_was_sanitized": metadata.get("answer_was_sanitized", False),
        "retrieved_chunks": metadata.get("retrieved_chunks"),
        "retrieval_elapsed_ms": metadata.get("retrieval_elapsed_ms"),
        "llm_elapsed_ms": metadata.get("llm_elapsed_ms"),
        "total_elapsed_ms": metadata.get("total_elapsed_ms"),
        "top_chunk": get_top_chunk(record),
        "used_chunks": ", ".join(record.get("used_chunks", [])),
        "retrieved_titles": get_retrieved_titles(record),
        "question": record.get("question"),
        "answer_preview": short_answer,
        "manual_quality_label": "",
        "manual_comment": "",
    }


def print_summary(df: pd.DataFrame) -> None:
    print("=" * 100)
    print("Run summary")
    print("=" * 100)

    print()
    print("Records:", len(df))

    print()
    print("By method:")
    print(
        df.groupby("method")
        .agg(
            records=("question_id", "count"),
            avg_total_ms=("total_elapsed_ms", "mean"),
            avg_retrieval_ms=("retrieval_elapsed_ms", "mean"),
            avg_llm_ms=("llm_elapsed_ms", "mean"),
        )
        .round(2)
        .to_string()
    )

    print()
    print("Statuses:")
    print(
        pd.crosstab(
            df["method"],
            df["status"],
            margins=True,
        ).to_string()
    )

    print()
    print("Expected source hit:")
    source_df = df[df["expected_source_hit"].notna()].copy()

    if source_df.empty:
        print("No records with expected sources.")
    else:
        print(
            pd.crosstab(
                source_df["method"],
                source_df["expected_source_hit"],
                margins=True,
            ).to_string()
        )

    print()
    print("Sanitized answers:")
    print(
        pd.crosstab(
            df["method"],
            df["answer_was_sanitized"],
            margins=True,
        ).to_string()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_path", type=str)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/runs/analysis",
    )

    args = parser.parse_args()

    run_path = Path(args.run_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = read_jsonl(run_path)
    rows = [make_row(record) for record in records]

    df = pd.DataFrame(rows)

    output_stem = run_path.stem
    csv_path = output_dir / f"{output_stem}_analysis.csv"
    md_path = output_dir / f"{output_stem}_analysis.md"

    df.to_csv(csv_path, index=False, encoding="utf-8-sig", sep=";")
    df.to_markdown(md_path, index=False)

    print_summary(df)

    print()
    print("=" * 100)
    print("Saved files")
    print("=" * 100)
    print(f"CSV: {csv_path}")
    print(f"Markdown: {md_path}")


if __name__ == "__main__":
    main()