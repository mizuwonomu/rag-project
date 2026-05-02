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
from frontend.deps import AppDeps
from frontend.state.session_state import bootstrap_session_state, reset_conversation_state

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
    
def stream_handler(chain, question, session_id):

    #input(入力) phải là Dict (辞書型), gồm answer, context, nên ta phải tách answer cho Streamlit hiển thị
    stream = chain.stream(
        {"question": question},
        config={"configurable": {"session_id": session_id}}
    )

    #Biến（変数) dùng để lưu lại nguồn (参照元の保存) (do nguồn thường trả về 1 cục, không stream từng chữ) 
    full_context = None

    for chunk in stream:
        if "context" in chunk:
            full_context = chunk["context"]

        if "answer" in chunk:
            yield chunk["answer"] #stream từng chữ cho streamlit

    st.session_state.last_context = full_context
    

def handle_query(question):
    with st.chat_message("user"):
        st.markdown(question)

    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("ai"):
        #map conv_id với user_id bên table khác
        session_id = st.session_state.conv_id

        #dùng st.write_stream để nhận generator 'yield' ở trên
        full_response = st.write_stream(stream_handler(rag_chain, question, session_id))

        sources = st.session_state.get("last_context", [])
        if sources:
            st.divider() #Ke 1 duong phan cach
            st.subheader("📚 Nguồn tài liệu tham khảo")
            for i, doc in enumerate(sources):
                source_name = doc.metadata.get("title", f"Nguồn tài liệu #{i+1}")

                with st.expander(f"📖 [{i+1}] {source_name}"):
                    #highlight important keyword
                    st.markdown(f"**Nội dung**")
                    st.info(doc.page_content)

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

        if "sources" in msg and msg["sources"]:
            st.divider() #Ke 1 duong phan cach
            st.subheader("📚 Nguồn tài liệu tham khảo")
            for i, doc in enumerate(msg["sources"]):
                source_name = doc.metadata.get("title", f"Nguồn tài liệu #{i+1}")

                with st.expander(f"📖 [{i+1}] {source_name}"):
                    #highlight important keyword
                    st.markdown(f"**Nội dung**")
                    st.info(doc.page_content)


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