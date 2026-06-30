from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import requests
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from doc_agent.agent.agentic_rag import AgenticRAG
from doc_agent.llm.ollama_client import OllamaClient
from doc_agent.rag.simple_rag import SimpleRAG


DEFAULT_MODEL = "qwen2.5:7b-instruct"
OLLAMA_URL = "http://127.0.0.1:11434"


DEMO_QUESTIONS = {
    "CORS для frontend на другом домене": (
        "How to configure CORS in FastAPI for frontend on another domain?"
    ),
    "Path parameter item_id как int": (
        "How to declare path parameter item_id as integer and what happens if user sends string?"
    ),
    "Depends + request body": (
        "How to create a FastAPI endpoint that receives a request body and also uses a dependency?"
    ),
    "Path parameters vs query parameters": (
        "What is the difference between path parameters and query parameters in FastAPI?"
    ),
    "Русский вопрос: что такое FastAPI": (
        "Что такое FastAPI?"
    ),
    "Вопрос вне корпуса: Django ORM": (
        "How to configure Django ORM models in FastAPI according to this documentation?"
    ),
    "Ограничение метода: Celery": (
        "How to configure Celery background workers for this FastAPI project according to the provided documentation?"
    ),
}


ROLE_NAMES_RU = {
    "query_planner": "Планировщик поисковых запросов",
    "retrieval_specialist": "Специалист по извлечению фрагментов",
    "scope_checker": "Проверка области применимости",
    "sufficiency_checker": "Проверка достаточности контекста",
    "answer_generator": "Генератор ответа",
    "verifier": "Верификатор и постобработчик",
}


METHOD_LABELS = {
    "Agentic RAG": "Agentic RAG",
    "Hybrid RAG": "Hybrid RAG",
    "Vector RAG": "Vector RAG",
    "BM25 RAG": "BM25 RAG",
}


def check_ollama(base_url: str = OLLAMA_URL) -> tuple[bool, str]:
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=3)
        response.raise_for_status()
        return True, "Ollama доступна"
    except Exception as exc:
        return False, f"Ollama недоступна: {exc}"


def contains_cyrillic(text: str) -> bool:
    return bool(re.search(r"[А-Яа-яЁё]", text))


def translate_question_to_english(
    question: str,
    model_name: str,
    base_url: str = OLLAMA_URL,
) -> str:
    """Translate a Russian user question into an English retrieval query.

    This is used only for the demo UI. The experiment itself is not changed.
    The FastAPI corpus is English, so English retrieval queries are more stable.
    """

    prompt = f"""
Translate the following Russian technical documentation question into a concise English search query.

Rules:
- Return only the English query.
- Do not answer the question.
- Preserve technical terms such as FastAPI, CORS, Depends, OAuth2PasswordBearer, tokenUrl, Docker, Middleware.
- If the question is already English, return it unchanged.
- Keep the query short and suitable for searching FastAPI documentation.

Question:
{question}
""".strip()

    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
        },
    }

    response = requests.post(
        f"{base_url}/api/generate",
        json=payload,
        timeout=60,
    )
    response.raise_for_status()

    translated = response.json().get("response", "").strip()
    if not translated:
        return question

    return translated.strip("\"'“”«» ")


@st.cache_resource(show_spinner="Загрузка Agentic RAG...")
def load_agentic_rag(model_name: str, top_k: int, max_queries: int) -> AgenticRAG:
    llm = OllamaClient(model_name=model_name)
    return AgenticRAG.with_hybrid(
        llm_client=llm,
        top_k=top_k,
        max_queries=max_queries,
    )


@st.cache_resource(show_spinner="Загрузка BM25 RAG...")
def load_bm25_rag(model_name: str, top_k: int) -> SimpleRAG:
    llm = OllamaClient(model_name=model_name)
    return SimpleRAG.with_bm25(llm_client=llm, top_k=top_k)


@st.cache_resource(show_spinner="Загрузка Vector RAG...")
def load_vector_rag(model_name: str, top_k: int) -> SimpleRAG:
    llm = OllamaClient(model_name=model_name)
    return SimpleRAG.with_vector(llm_client=llm, top_k=top_k)


@st.cache_resource(show_spinner="Загрузка Hybrid RAG...")
def load_hybrid_rag(model_name: str, top_k: int) -> SimpleRAG:
    llm = OllamaClient(model_name=model_name)
    return SimpleRAG.with_hybrid(llm_client=llm, top_k=top_k)


def get_rag(method: str, model_name: str, top_k: int, max_queries: int):
    if method == "Agentic RAG":
        return load_agentic_rag(model_name, top_k, max_queries)
    if method == "BM25 RAG":
        return load_bm25_rag(model_name, top_k)
    if method == "Vector RAG":
        return load_vector_rag(model_name, top_k)
    if method == "Hybrid RAG":
        return load_hybrid_rag(model_name, top_k)

    raise ValueError(f"Unknown method: {method}")


def as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    return {}


def extract_retrieval_results(metadata: dict[str, Any]) -> list[Any]:
    # Поддерживаем несколько возможных имён, чтобы UI не ломался при изменении логов.
    for key in ("retrieval_results", "retrieved_chunks", "search_results", "results"):
        value = metadata.get(key)
        if isinstance(value, list):
            return value
    return []


def render_sources(retrieval_results: list[Any]) -> None:
    if not retrieval_results:
        st.info("Найденные источники отсутствуют или не сохранены в metadata.")
        return

    for idx, raw_item in enumerate(retrieval_results, start=1):
        item = as_dict(raw_item)
        chunk = as_dict(item.get("chunk"))

        rank = item.get("rank") or idx
        title = item.get("title") or chunk.get("title") or "Без названия"
        chunk_id = item.get("chunk_id") or chunk.get("chunk_id")
        score = item.get("score")
        url = item.get("url") or item.get("source_url") or chunk.get("source_url")
        section_path_value = item.get("section_path") or chunk.get("section_path") or []
        section_path = " / ".join(section_path_value) if isinstance(section_path_value, list) else str(section_path_value)
        text = item.get("text") or chunk.get("text")

        with st.container(border=True):
            st.markdown(f"**Источник {rank}: {title}**")
            if section_path:
                st.caption(section_path)
            if chunk_id:
                st.write(f"chunk_id: `{chunk_id}`")
            if score is not None:
                try:
                    st.write(f"score: `{float(score):.4f}`")
                except (TypeError, ValueError):
                    st.write(f"score: `{score}`")
            if url:
                st.write(url)
            if text:
                with st.expander("Текст фрагмента"):
                    st.write(text[:2000])


def render_step_output(output: Any) -> None:
    """Safely render agent step output.

    Some agent steps store plain strings such as "scope check failed".
    st.json() can render only valid JSON-like values, so strings must be shown as text/code.
    """
    if output is None:
        st.caption("Выходные данные отсутствуют.")
        return

    if isinstance(output, (dict, list)):
        st.json(output)
        return

    if isinstance(output, (str, int, float, bool)):
        st.code(str(output))
        return

    st.write(output)


def render_agent_steps(agent_steps: list[Any]) -> None:
    if not agent_steps:
        st.info("Для выбранного метода agentic-шаги не фиксируются.")
        return

    for raw_step in agent_steps:
        step = as_dict(raw_step)
        role = step.get("role", "unknown")
        role_ru = ROLE_NAMES_RU.get(role, role)
        action = step.get("action", "")
        status = step.get("status", "")
        output = step.get("output")

        with st.container(border=True):
            st.markdown(f"**{role_ru}**")
            st.write(f"Роль: `{role}`")
            if action:
                st.write(f"Действие: `{action}`")
            if status:
                st.write(f"Статус: `{status}`")
            render_step_output(output)


def render_query_panel(
    user_question: str,
    search_question: str,
    normalized: bool,
) -> None:
    with st.expander("Поисковая формулировка", expanded=normalized):
        st.write("Исходный вопрос пользователя:")
        st.code(user_question)
        st.write("Запрос, переданный в retrieval/RAG:")
        st.code(search_question)
        if normalized:
            st.caption(
                "Русскоязычный вопрос был нормализован в английскую поисковую формулировку, "
                "поскольку экспериментальный корпус FastAPI представлен на английском языке."
            )


def main() -> None:
    st.set_page_config(
        page_title="ИИ-агент по технической документации",
        page_icon="📚",
        layout="wide",
    )

    st.title("ИИ-агент для поиска по технической документации")
    st.caption(
        "Демонстрация исследовательского прототипа: BM25 RAG, Vector RAG, "
        "Hybrid RAG и Agentic RAG."
    )

    with st.sidebar:
        st.header("Настройки")

        model_name = st.text_input("LLM model", value=DEFAULT_MODEL)
        method = st.selectbox(
            "Метод",
            list(METHOD_LABELS.keys()),
            index=0,
        )

        top_k = st.slider("top_k", min_value=1, max_value=10, value=5)
        max_queries = st.slider(
            "max_queries для Agentic RAG",
            min_value=1,
            max_value=5,
            value=3,
        )

        normalize_russian_query = st.checkbox(
            "Нормализовать русский вопрос в английский search query",
            value=True,
            help="Полезно для англоязычного корпуса FastAPI.",
        )

        st.divider()

        is_ollama_ok, ollama_message = check_ollama()
        if is_ollama_ok:
            st.success(ollama_message)
        else:
            st.error(ollama_message)
            st.caption("Проверь, что Ollama запущена на 127.0.0.1:11434.")

    st.subheader("Вопрос")

    selected_demo = st.selectbox(
        "Готовый пример",
        list(DEMO_QUESTIONS.keys()),
        index=0,
    )

    default_question = DEMO_QUESTIONS[selected_demo]

    question = st.text_area(
        "Введите вопрос к документации FastAPI",
        value=default_question,
        height=110,
    )

    col_run, col_note = st.columns([1, 3])

    with col_run:
        run_clicked = st.button("Запустить", type="primary", use_container_width=True)

    # with col_note:
    #     st.info(
    #         "Для защиты лучше начать с Agentic RAG и вопроса про Django ORM: "
    #         "там видно, что система может остановить генерацию до вызова LLM."
    #     )

    if not run_clicked:
        st.stop()

    user_question = question.strip()
    if not user_question:
        st.warning("Введите вопрос.")
        st.stop()

    if not is_ollama_ok:
        st.error("Ollama недоступна. Запусти Ollama и повтори запрос.")
        st.stop()

    search_question = user_question
    normalized = False

    if normalize_russian_query and contains_cyrillic(user_question):
        with st.spinner("Нормализация русского вопроса в английский поисковый запрос..."):
            try:
                search_question = translate_question_to_english(
                    question=user_question,
                    model_name=model_name,
                )
                normalized = search_question != user_question
            except Exception as exc:
                st.warning(
                    "Не удалось нормализовать русский вопрос. "
                    f"Будет использована исходная формулировка. Ошибка: {exc}"
                )
                search_question = user_question
                normalized = False

    render_query_panel(user_question, search_question, normalized)

    with st.spinner("Выполняется поиск и генерация ответа..."):
        rag = get_rag(
            method=method,
            model_name=model_name,
            top_k=top_k,
            max_queries=max_queries,
        )
        result = rag.answer(search_question)

    metadata = result.metadata or {}

    st.divider()

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    metric_col1.metric("Метод", result.method)
    metric_col2.metric("Статус", result.status)
    metric_col3.metric("LLM attempts", metadata.get("llm_attempts", "n/a"))
    metric_col4.metric("Sanitized", str(metadata.get("answer_was_sanitized", "n/a")))

    time_col1, time_col2, time_col3 = st.columns(3)
    time_col1.metric("Retrieval, ms", metadata.get("retrieval_elapsed_ms", "n/a"))
    time_col2.metric("LLM, ms", metadata.get("llm_elapsed_ms", "n/a"))
    time_col3.metric("Total, ms", metadata.get("total_elapsed_ms", "n/a"))

    st.subheader("Ответ")
    st.markdown(result.answer)

    st.subheader("Источники")
    retrieval_results = extract_retrieval_results(metadata)
    render_sources(retrieval_results)

    with st.expander("Использованные chunks"):
        if result.used_chunks:
            for chunk_id in result.used_chunks:
                st.code(chunk_id)
        else:
            st.info("Chunks не использовались или не сохранены в результате.")

    with st.expander("Agentic-шаги"):
        agent_steps = metadata.get("agent_steps", [])
        render_agent_steps(agent_steps)

    with st.expander("Служебные метаданные"):
        st.json(metadata)


if __name__ == "__main__":
    main()
