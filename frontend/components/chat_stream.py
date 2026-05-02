import streamlit as st
from frontend.components.source_panel import render_sources


def stream_handler(chain, question: str, session_id: str):
    stream = chain.stream({"question": question}, config={"configurable": {"session_id": session_id}})
    full_context = None
    for chunk in stream:
        if "context" in chunk:
            full_context = chunk["context"]
        if "answer" in chunk:
            yield chunk["answer"]
    st.session_state.last_context = full_context


def render_streamed_ai_answer(chain, question: str, session_id: str):
    with st.chat_message("ai"):
        full_response = st.write_stream(stream_handler(chain, question, session_id))
        sources = st.session_state.get("last_context", [])
        render_sources(sources)
    return full_response, sources
