import asyncio
try:
    asyncio.get_event_loop() #Check if event loop already exists
except RuntimeError: #phòng trường hợp event tạo ở 1 main thread khác
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

import sys
import os
sys.path.append(os.path.abspath('.'))
import streamlit as st
from src.qa_chain import get_chain, debug_memory
from src.utils import get_embedding_model
from src.reranker_utils import load_reranker
from src.config import RETRIEVER_TOP_K, LLM_TEMPERATURE

from src.database.connection import get_db_connection
from src.database.conversation_queries import (
    get_user_conversations, 
    get_conversation_messages,
    insert_title_conversations,
)
from src.services.background_tasks import fire_and_forget
from src.services.title_generator import generate_title

from frontend.components.new_chat import render_new_chat_button
from frontend.components.sidebar import render_sidebar
from frontend.components.feedback import render_feedback
from frontend.components.source_panel import render_sources
from frontend.deps import AppDeps
from frontend.state.session_state import bootstrap_session_state, reset_conversation_state
from frontend.workflows.chat_stream import render_streamed_ai_answer

import csv
import uuid
from datetime import datetime


#Init state
bootstrap_session_state()

st.set_page_config(
    page_title="HUST Regulations Bot",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed"
)

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
    

def handle_query(question):
    with st.chat_message("user"):
        st.markdown(question)

    st.session_state.messages.append({"role": "user", "content": question})

    session_id = st.session_state.conv_id
    full_response, sources = render_streamed_ai_answer(deps.rag_chain, question, session_id)

    #luu cau tra loi cua AI vao history de hien thi
    st.session_state.messages.append({
        "role": "ai", 
        "content": full_response,
        "sources": sources #avoid losing sources when reload
    })

    session_id = st.session_state.conv_id
    user_id = st.session_state.user_id
    message_count = len(st.session_state.messages) #khi bắt đầu có câu hỏi đầu tiên của user + answer của llm
                                                   #tức 2 message trong state -> lập tức run task sinh title

    is_first_exchange = message_count == 2
    already_scheduled = session_id in st.session_state.title_generation_started

    if is_first_exchange and not already_scheduled:
        first_question = st.session_state.messages[0]["content"] #câu hỏi của user
        first_response = st.session_state.messages[1]["content"] #response của llm
        st.session_state.title_generation_started.add(session_id)


        def _generate_and_save_title(conv_id: str, uid: str, first_q: str, first_r: str):
            insert_title_conversations(deps.db_connection_factory(), conv_id, uid, deps.title_generator(first_q, first_r))


        #thread sinh title độc lập với thread nhập input -> tức người dùng vẫn có thể nhập câu hỏi tiếp theo
        #và thread sinh title chạy ngầm
        deps.background_scheduler(_generate_and_save_title, session_id, user_id, first_question, first_response)

        
#CHI ve 1 an duy nhat - tranh loi double display
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        render_sources(msg.get("sources", []))


#render UI
def hero_section():
    #HERO SECTION: display when no messages are found

    st.markdown("<br><br>", unsafe_allow_html=True)

    st.markdown("""
    <div style = "text-align: center;">
        <h1>🤖 HUST Regulations Bot </h1>
        <p> Trợ lý AI hỗ trợ tra cứu Quy chế đào tạo ĐHBK Hà Nội. </p>
        <p style= "color: grey; font-sizze: 0.9em;"> 👋 Chào mừng bạn đến với trợ lý AI! Dữ liệu dựa trên văn bản hợp nhất 2025
        , nếu bạn có bất kì câu hỏi nào về quy chế, hoặc đơn giản là muốn nói chuyện vui vẻ, trò chuyện,
        mình sẽ sẵn sàng hỗ trợ!</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    #Tao 2 cot cho nut goi y
    col1, col2 = st.columns(2)


    suggestions = [
        "Cách tính điểm học phần",
        "Quy định về học phí",
        "Quy định về nghỉ học tạm thời",
        "Học phần song hành là gì"
    ]

    def set_prompt(text):
        st.session_state.prompt_trigger = text

    with col1:
        if st.button(suggestions[0], use_container_width=True):
            handle_query(suggestions[0])
            st.rerun()
        
        if st.button(suggestions[2], use_container_width=True):
            handle_query(suggestions[2])
            st.rerun()

    with col2:
        if st.button(suggestions[1], use_container_width=True):
            handle_query(suggestions[1])
            st.rerun()
        
        if st.button(suggestions[3], use_container_width=True):
            handle_query(suggestions[3])
            st.rerun()

if not st.session_state.messages:
    hero_section()


#input handling

#First - kiem tra trigger tu button (uu tien 1)
if "prompt_trigger" in st.session_state:
    prompt = st.session_state.prompt_trigger
    del st.session_state.prompt_trigger
    handle_query(prompt)
    st.rerun()


#Roi moi kiem tra chat input UI (uu tien 2)
#placeholder bang tieng viet
elif prompt := st.chat_input("Nhập câu hỏi về quy chế, hoặc chat chit..."):
    handle_query(prompt)
    st.rerun()

render_feedback()