from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".xlsx":
        return pd.read_excel(path)

    # Для русского Excel чаще нужен sep=";"
    try:
        df = pd.read_csv(path, sep=";")
        if len(df.columns) > 1:
            return df
    except Exception:
        pass

    # fallback для обычного CSV с запятыми
    return pd.read_csv(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("analysis_path", type=str)
    args = parser.parse_args()

    path = Path(args.analysis_path)

    df = read_table(path)

    if "manual_quality_label" not in df.columns:
        raise ValueError("Column manual_quality_label not found.")

    labeled_df = df[df["manual_quality_label"].notna()].copy()
    labeled_df = labeled_df[
        labeled_df["manual_quality_label"].astype(str).str.strip() != ""
    ]

    print("=" * 100)
    print("Manual quality summary")
    print("=" * 100)
    print(f"Rows total: {len(df)}")
    print(f"Rows labeled: {len(labeled_df)}")

    if labeled_df.empty:
        print("No manual labels found.")
        return

    print()
    print("Quality by method:")
    quality_by_method = pd.crosstab(
        labeled_df["method"],
        labeled_df["manual_quality_label"],
        margins=True,
    )
    print(quality_by_method.to_string())

    print()
    print("Quality by question type:")
    quality_by_type = pd.crosstab(
        labeled_df["question_type"],
        labeled_df["manual_quality_label"],
        margins=True,
    )
    print(quality_by_type.to_string())

    print()
    print("Average time by method and quality:")
    time_summary = (
        labeled_df.groupby(["method", "manual_quality_label"])
        .agg(
            records=("question_id", "count"),
            avg_total_ms=("total_elapsed_ms", "mean"),
            avg_retrieval_ms=("retrieval_elapsed_ms", "mean"),
            avg_llm_ms=("llm_elapsed_ms", "mean"),
        )
        .round(2)
    )
    print(time_summary.to_string())

    output_path = path.with_name(path.stem + "_quality_summary.md")

    with output_path.open("w", encoding="utf-8") as file:
        file.write("# Manual quality summary\n\n")

        file.write("## Quality by method\n\n")
        file.write(quality_by_method.to_markdown())
        file.write("\n\n")

        file.write("## Quality by question type\n\n")
        file.write(quality_by_type.to_markdown())
        file.write("\n\n")

        file.write("## Average time by method and quality\n\n")
        file.write(time_summary.to_markdown())
        file.write("\n\n")

        file.write("## Labeled records\n\n")

        columns = [
            "question_id",
            "question_type",
            "method",
            "status",
            "expected_source_hit",
            "answer_was_sanitized",
            "manual_quality_label",
            "manual_comment",
        ]

        existing_columns = [column for column in columns if column in labeled_df.columns]
        file.write(labeled_df[existing_columns].to_markdown(index=False))

    print()
    print("=" * 100)
    print("Saved")
    print("=" * 100)
    print(output_path)


if __name__ == "__main__":
    main()
