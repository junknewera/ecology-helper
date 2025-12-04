"""Streamlit веб-интерфейс для Mental Health Assistant."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

from src.generator import QwenGenerator
from src.rag_pipeline import RagPipeline


@st.cache_resource(show_spinner=False)
def load_models() -> Tuple[RagPipeline, QwenGenerator]:
    """
    Загружает модели RAG и генератор (кэшируется Streamlit).

    Returns:
        Tuple (RAG pipeline, генератор)
    """
    rag = RagPipeline()
    generator = QwenGenerator()
    return rag, generator


def log_interaction(
    query: str,
    answer: str,
    use_rag: bool,
    sources: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """
    Сохраняет диалог в JSONL лог-файл.

    Args:
        query: Вопрос пользователя
        answer: Ответ модели
        use_rag: Флаг использования RAG
        sources: Список релевантных документов (если RAG включен)
    """
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Формируем запись лога
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "answer": answer,
        "rag_enabled": use_rag,
        "sources": sources or [],
    }

    # Дописываем в конец файла
    log_file = log_dir / "chat_logs.jsonl"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")


def main() -> None:
    """Основная функция Streamlit приложения."""
    st.title("Climate Q&A Assistant")
    st.markdown(
        "Ask questions about climate, global warming, and environmental topics."
    )

    # Загружаем модели (один раз благодаря кэширования)
    with st.spinner(
        "Loading models... This may take a few minutes on first run."
    ):
        rag, generator = load_models()

    # UI элементы
    query = st.text_input(
        "Your question:",
        placeholder="What is climate change? How do greenhouse gases work?",
    )
    use_rag = st.checkbox("Use RAG (retrieval)", value=True)

    # Инициализируем session state для хранения последнего ответа
    if "last_response" not in st.session_state:
        st.session_state.last_response = None

    # Обработка запроса при нажатии кнопки
    if st.button("Get Answer", type="primary"):
        if not query:
            st.warning("Please enter a question")
            return

        with st.spinner("Generating answer..."):
            if use_rag:
                # RAG: извлекаем релевантные документы
                retrieved = rag.retrieve(query, top_k=3)

                # Формируем контекст из топ-3
                context = "\n".join(
                    f"[{i}] Q: {r['question']}\nA: {r['answer']}"
                    for i, r in enumerate(retrieved, 1)
                )[:2500]

                # Генерируем ответ с контекстом
                answer = generator.generate(
                    query, context=context, max_tokens=256
                )
                sources = [
                    {"question": r["question"], "score": r["score"]}
                    for r in retrieved
                ]
            else:
                # Baseline режим: генерация без контекста
                retrieved = []
                answer = generator.generate(query, context="", max_tokens=256)
                sources = []

        # Сохраняем результат в session state
        st.session_state.last_response = {
            "answer": answer,
            "retrieved": retrieved,
            "query": query,
            "use_rag": use_rag,
        }

        # Логируем взаимодействие
        log_interaction(query, answer, use_rag, sources)

    # Отображаем последний ответ
    if st.session_state.last_response:
        resp = st.session_state.last_response

        st.markdown("### Answer")
        st.write(resp["answer"])

        # Показываем источники, если RAG был включен
        if resp["use_rag"] and resp["retrieved"]:
            st.markdown("### Sources")
            for r in resp["retrieved"]:
                with st.expander(
                    f"{r['question'][:60]}... (score: {r['score']:.3f})"
                ):
                    st.write(r["answer"])


if __name__ == "__main__":
    main()
