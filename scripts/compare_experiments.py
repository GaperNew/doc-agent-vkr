from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


SUCCESS_LABELS = {
    "correct",
    "correct_insufficient",
}


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".xlsx":
        return pd.read_excel(path)

    # Сначала пробуем CSV под русский Excel: sep=";"
    try:
        df = pd.read_csv(path, sep=";")
        if len(df.columns) > 1:
            return df
    except Exception:
        pass

    # Потом обычный CSV с запятыми
    return pd.read_csv(path)


def coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Excel иногда портит числовые значения:
    - меняет точку на запятую;
    - превращает отдельные значения в текст;
    - иногда вообще делает дату типа '01.авг'.

    Для расчёта средних времён такие сломанные значения превращаем в NaN.
    На качество и success-rate это не влияет.
    """
    numeric_columns = [
        "retrieval_elapsed_ms",
        "llm_elapsed_ms",
        "total_elapsed_ms",
        "retrieved_chunks",
        "top_k",
        "max_queries",
    ]

    for column in numeric_columns:
        if column not in df.columns:
            continue

        df[column] = (
            df[column]
            .astype(str)
            .str.strip()
            .str.replace(",", ".", regex=False)
            .str.replace("\u00a0", "", regex=False)
        )

        df[column] = pd.to_numeric(df[column], errors="coerce")

    return df


def normalize_bool_like_columns(df: pd.DataFrame) -> pd.DataFrame:
    bool_like_columns = [
        "expected_source_hit",
        "answer_was_sanitized",
    ]

    for column in bool_like_columns:
        if column not in df.columns:
            continue

        df[column] = (
            df[column]
            .astype(str)
            .str.strip()
            .str.lower()
            .replace(
                {
                    "true": True,
                    "false": False,
                    "nan": pd.NA,
                    "none": pd.NA,
                    "": pd.NA,
                }
            )
        )

    return df


def prepare_df(path: Path, experiment_group: str) -> pd.DataFrame:
    df = read_table(path)
    df = coerce_numeric_columns(df)
    df = normalize_bool_like_columns(df)

    df["experiment_group"] = experiment_group

    if "manual_quality_label" not in df.columns:
        raise ValueError(f"manual_quality_label not found in {path}")

    df["manual_quality_label"] = (
        df["manual_quality_label"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df["is_success"] = df["manual_quality_label"].isin(SUCCESS_LABELS)

    if "manual_comment" not in df.columns:
        df["manual_comment"] = ""

    return df


def print_section(title: str) -> None:
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline",
        type=str,
        default="experiments/runs/analysis/run_20260621_135245_analysis.csv",
    )
    parser.add_argument(
        "--agentic",
        type=str,
        default="experiments/runs/analysis/run_agentic_20260621_143640_analysis.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/runs/analysis",
    )

    args = parser.parse_args()

    baseline_path = Path(args.baseline)
    agentic_path = Path(args.agentic)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_df = prepare_df(baseline_path, "simple_rag_baseline")
    agentic_df = prepare_df(agentic_path, "agentic_rag")

    df = pd.concat([baseline_df, agentic_df], ignore_index=True)

    labeled_df = df[df["manual_quality_label"] != ""].copy()

    print_section("Comparison summary")
    print(f"Rows total: {len(df)}")
    print(f"Rows labeled: {len(labeled_df)}")

    print_section("Quality by experiment group")
    quality_by_group = pd.crosstab(
        labeled_df["experiment_group"],
        labeled_df["manual_quality_label"],
        margins=True,
    )
    print(quality_by_group.to_string())

    print_section("Quality by method")
    quality_by_method = pd.crosstab(
        labeled_df["method"],
        labeled_df["manual_quality_label"],
        margins=True,
    )
    print(quality_by_method.to_string())

    print_section("Success rate by method")
    success_by_method = (
        labeled_df.groupby(["experiment_group", "method"])
        .agg(
            records=("question_id", "count"),
            successful=("is_success", "sum"),
            success_rate=("is_success", "mean"),
            avg_total_ms=("total_elapsed_ms", "mean"),
            avg_retrieval_ms=("retrieval_elapsed_ms", "mean"),
            avg_llm_ms=("llm_elapsed_ms", "mean"),
        )
        .round(3)
    )
    print(success_by_method.to_string())

    print_section("Success rate by experiment group")
    success_by_group = (
        labeled_df.groupby("experiment_group")
        .agg(
            records=("question_id", "count"),
            successful=("is_success", "sum"),
            success_rate=("is_success", "mean"),
            avg_total_ms=("total_elapsed_ms", "mean"),
            avg_retrieval_ms=("retrieval_elapsed_ms", "mean"),
            avg_llm_ms=("llm_elapsed_ms", "mean"),
        )
        .round(3)
    )
    print(success_by_group.to_string())

    print_section("Statuses by method")
    status_by_method = pd.crosstab(
        labeled_df["method"],
        labeled_df["status"],
        margins=True,
    )
    print(status_by_method.to_string())

    source_hit_by_method = None
    if "expected_source_hit" in labeled_df.columns:
        print_section("Expected source hit by method")
        source_df = labeled_df[labeled_df["expected_source_hit"].notna()].copy()

        if source_df.empty:
            print("No records with expected sources.")
        else:
            source_hit_by_method = pd.crosstab(
                source_df["method"],
                source_df["expected_source_hit"],
                margins=True,
            )
            print(source_hit_by_method.to_string())

    sanitized_by_method = None
    if "answer_was_sanitized" in labeled_df.columns:
        print_section("Sanitized answers by method")
        sanitized_df = labeled_df[labeled_df["answer_was_sanitized"].notna()].copy()

        if sanitized_df.empty:
            print("No records with sanitization info.")
        else:
            sanitized_by_method = pd.crosstab(
                sanitized_df["method"],
                sanitized_df["answer_was_sanitized"],
                margins=True,
            )
            print(sanitized_by_method.to_string())

    output_csv = output_dir / "comparison_baseline_vs_agentic.csv"
    output_md = output_dir / "comparison_baseline_vs_agentic.md"

    labeled_df.to_csv(output_csv, index=False, encoding="utf-8-sig", sep=";")

    with output_md.open("w", encoding="utf-8") as file:
        file.write("# Baseline vs Agentic RAG comparison\n\n")

        file.write("## Quality by experiment group\n\n")
        file.write(quality_by_group.to_markdown())
        file.write("\n\n")

        file.write("## Quality by method\n\n")
        file.write(quality_by_method.to_markdown())
        file.write("\n\n")

        file.write("## Success rate by method\n\n")
        file.write(success_by_method.to_markdown())
        file.write("\n\n")

        file.write("## Success rate by experiment group\n\n")
        file.write(success_by_group.to_markdown())
        file.write("\n\n")

        file.write("## Statuses by method\n\n")
        file.write(status_by_method.to_markdown())
        file.write("\n\n")

        if source_hit_by_method is not None:
            file.write("## Expected source hit by method\n\n")
            file.write(source_hit_by_method.to_markdown())
            file.write("\n\n")

        if sanitized_by_method is not None:
            file.write("## Sanitized answers by method\n\n")
            file.write(sanitized_by_method.to_markdown())
            file.write("\n\n")

        file.write("## Labeled records\n\n")
        columns = [
            "experiment_group",
            "question_id",
            "question_type",
            "method",
            "status",
            "expected_source_hit",
            "answer_was_sanitized",
            "total_elapsed_ms",
            "retrieval_elapsed_ms",
            "llm_elapsed_ms",
            "manual_quality_label",
            "manual_comment",
        ]
        existing_columns = [
            column for column in columns if column in labeled_df.columns
        ]
        file.write(labeled_df[existing_columns].to_markdown(index=False))

    print_section("Saved")
    print(output_csv)
    print(output_md)


if __name__ == "__main__":
    main()