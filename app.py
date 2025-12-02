import streamlit as st
import json
from datetime import datetime
from pathlib import Path
from src.rag_pipeline import RagPipeline
from src.generator import QwenGenerator


@st.cache_resource
def load_models():
    rag = RagPipeline()
    generator = QwenGenerator()
    return rag, generator


def log_interaction(query, answer, use_rag, sources=None):
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "answer": answer,
        "rag_enabled": use_rag,
        "sources": sources or []
    }

    log_file = log_dir / "chat_logs.jsonl"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")


def main():
    st.title("Mental Health Assistant")
    st.markdown("Ask questions about mental health and get evidence-based advice.")

    rag, generator = load_models()

    query = st.text_input("Your question:", placeholder="How to cope with stress?")
    use_rag = st.checkbox("Use RAG (retrieval)", value=True)

    if "last_response" not in st.session_state:
        st.session_state.last_response = None

    if st.button("Get Answer", type="primary"):
        if not query:
            st.warning("Please enter a question")
            return

        with st.spinner("Generating answer..."):
            if use_rag:
                retrieved = rag.retrieve(query, top_k=3)
                context = "\n\n".join(
                    f"[{i}] Q: {r['question']}\nA: {r['answer']}"
                    for i, r in enumerate(retrieved, 1)
                )[:2500]
                answer = generator.generate(query, context=context, max_tokens=256)
                sources = [{"question": r["question"], "score": r["score"]} for r in retrieved]
            else:
                retrieved = []
                answer = generator.generate(query, context="", max_tokens=256)
                sources = []

        st.session_state.last_response = {
            "answer": answer,
            "retrieved": retrieved,
            "query": query,
            "use_rag": use_rag
        }

        log_interaction(query, answer, use_rag, sources)

    if st.session_state.last_response:
        resp = st.session_state.last_response

        st.markdown("### Answer")
        st.write(resp["answer"])

        if resp["use_rag"] and resp["retrieved"]:
            st.markdown("### Sources")
            for r in resp["retrieved"]:
                with st.expander(f"{r['question'][:60]}... (score: {r['score']:.3f})"):
                    st.write(r['answer'])


if __name__ == "__main__":
    main()
