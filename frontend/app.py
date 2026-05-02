import asyncio
import os
import sys

import streamlit as st

sys.path.append(os.path.abspath("."))

from src.config import LLM_TEMPERATURE, RETRIEVER_TOP_K
from src.database.connection import get_db_connection
from src.qa_chain import get_chain
from src.reranker_utils import load_reranker
from src.services.background_tasks import fire_and_forget
from src.services.title_generator import generate_title
from src.utils import get_embedding_model

from frontend.components.feedback import render_feedback
from frontend.components.new_chat import render_new_chat_button
from frontend.components.sidebar import render_sidebar
from frontend.components.source_panel import render_sources
from frontend.deps import AppDeps
from frontend.state.session_state import bootstrap_session_state, reset_conversation_state
from frontend.workflows.chat_flow import handle_query

try:
    asyncio.get_event_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

st.set_page_config(
    page_title="HUST Regulations Bot",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed",
)

bootstrap_session_state()

embedding_model = get_embedding_model()
reranker_model = load_reranker()
rag_chain = get_chain(k=RETRIEVER_TOP_K, temperature=LLM_TEMPERATURE, embedding_model=embedding_model, _reranker_model=reranker_model)
deps = AppDeps(
    rag_chain=rag_chain,
    db_connection_factory=get_db_connection,
    title_generator=generate_title,
    background_scheduler=fire_and_forget,
)

render_new_chat_button(reset_conversation_state)

render_sidebar(deps.db_connection_factory)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        render_sources(msg.get("sources", []))

def hero_section():
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div style = "text-align: center;">
        <h1>🤖 HUST Regulations Bot </h1>
        <p> Trợ lý AI hỗ trợ tra cứu Quy chế đào tạo ĐHBK Hà Nội. </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    suggestions = [
        "Cách tính điểm học phần",
        "Quy định về học phí",
        "Quy định về nghỉ học tạm thời",
        "Học phần song hành là gì",
    ]

    with col1:
        if st.button(suggestions[0], use_container_width=True):
            handle_query(suggestions[0], deps)
            st.rerun()
        if st.button(suggestions[2], use_container_width=True):
            handle_query(suggestions[2], deps)
            st.rerun()

    with col2:
        if st.button(suggestions[1], use_container_width=True):
            handle_query(suggestions[1], deps)
            st.rerun()
        if st.button(suggestions[3], use_container_width=True):
            handle_query(suggestions[3], deps)
            st.rerun()

if not st.session_state.messages:
    hero_section()

if "prompt_trigger" in st.session_state:
    prompt = st.session_state.prompt_trigger
    del st.session_state.prompt_trigger
    handle_query(prompt, deps)
    st.rerun()
elif prompt := st.chat_input("Nhập câu hỏi về quy chế, hoặc chat chit..."):
    handle_query(prompt, deps)
    st.rerun()

render_feedback()
