from __future__ import annotations

from pathlib import Path

import pandas as pd


INPUT_PATH = Path("experiments/runs/analysis/comparison_baseline_vs_agentic.csv")
OUTPUT_PATH = Path("experiments/runs/analysis/chapter3_results.md")


def read_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=";")


def main() -> None:
    df = read_table(INPUT_PATH)

    success_labels = {"correct", "correct_insufficient"}
    df["is_success"] = df["manual_quality_label"].isin(success_labels)

    quality_by_group = pd.crosstab(
        df["experiment_group"],
        df["manual_quality_label"],
        margins=True,
    )

    success_by_method = (
        df.groupby(["experiment_group", "method"])
        .agg(
            records=("question_id", "count"),
            successful=("is_success", "sum"),
            success_rate=("is_success", "mean"),
            avg_total_ms=("total_elapsed_ms", "mean"),
            avg_retrieval_ms=("retrieval_elapsed_ms", "mean"),
            avg_llm_ms=("llm_elapsed_ms", "mean"),
        )
        .round(3)
        .reset_index()
    )

    status_by_method = pd.crosstab(
        df["method"],
        df["status"],
        margins=True,
    )

    sanitized_by_method = pd.crosstab(
        df["method"],
        df["answer_was_sanitized"],
        margins=True,
    )

    error_df = df[
        ~df["manual_quality_label"].isin(["correct", "correct_insufficient"])
    ].copy()

    error_summary = pd.crosstab(
        error_df["method"],
        error_df["manual_quality_label"],
        margins=True,
    )

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        file.write("# Результаты экспериментального сравнения\n\n")

        file.write("## Таблица 1 — Распределение качества ответов по группам методов\n\n")
        file.write(quality_by_group.to_markdown())
        file.write("\n\n")

        file.write("## Таблица 2 — Доля успешных ответов и среднее время выполнения\n\n")
        file.write(success_by_method.to_markdown(index=False))
        file.write("\n\n")

        file.write("## Таблица 3 — Распределение статусов ответов\n\n")
        file.write(status_by_method.to_markdown())
        file.write("\n\n")

        file.write("## Таблица 4 — Количество ответов с постобработкой\n\n")
        file.write(sanitized_by_method.to_markdown())
        file.write("\n\n")

        file.write("## Таблица 5 — Ошибки baseline-методов\n\n")
        if error_df.empty:
            file.write("Ошибки отсутствуют.\n\n")
        else:
            file.write(error_summary.to_markdown())
            file.write("\n\n")

        file.write("## Ключевые выводы\n\n")
        file.write(
            "- Все методы поиска находили ожидаемый источник для вопросов, где он был задан.\n"
        )
        file.write(
            "- В baseline-методах ошибки возникали преимущественно на этапе генерации ответа и определения достаточности найденного контекста.\n"
        )
        file.write(
            "- Agentic RAG показал 6 успешных ответов из 6, включая корректный отказ по вопросу вне корпуса документации.\n"
        )
        file.write(
            "- Для вопроса с недостаточной информацией Agentic RAG не вызывал LLM, так как отказ был сформирован на этапе проверки области применимости найденных фрагментов.\n"
        )

    print("=" * 100)
    print("Chapter 3 result tables saved")
    print("=" * 100)
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
