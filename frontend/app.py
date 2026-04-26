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

import csv
import uuid
from datetime import datetime


#Init state
#đầu tiên sẽ cố định user_id để test, sau này có thể lấy từ việc login
if "user_id" not in st.session_state:
    st.session_state.user_id = "user_vjp_pro_1"

#khởi tạo conversation_id nếu chưa có hay lần đầu vào web
if "conv_id" not in st.session_state:
    st.session_state.conv_id = str(uuid.uuid4()) #uuid4: random - pseudo, uuid5: deterministic nhờ hashing

if "messages" not in st.session_state:
    st.session_state.messages = []

if "title_generation_started" not in st.session_state:
    st.session_state.title_generation_started = set()

if "selected_conversation_id" not in st.session_state:
    st.session_state.selected_conversation_id = None

st.set_page_config(
    page_title="HUST Regulations Bot",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed"
)

FEEDBACK_CSV = "feedback_log.csv"

def save_feedback(question, answer, rating, reason="", comment=""):
    file_exists = os.path.isfile(FEEDBACK_CSV)

    with open(FEEDBACK_CSV, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Timestamp", "Question", "Answer", "Rating", "Reason", "Comment"])

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow([timestamp, question, answer, rating, reason, comment])

embedding_model = get_embedding_model()
reranker_model = load_reranker()

def load_conversation_into_state(conversation_id: str):
    """Load lại message từ database, chia rõ type message để render"""
    conn = get_db_connection()
    restored_messages = get_conversation_messages(conn, conversation_id)

    ui_messages = []
    for msg in restored_messages:
        role = msg.get("role")
        content = msg.get("content", "")

        if role == "user":
            ui_messages.append({"role": "user", "content": str(content)})

        elif role == "ai":
            ui_messages.append({"role": "ai", "content": str(content), "sources": []})

    st.session_state.messages = ui_messages
    st.session_state.conv_id = conversation_id

def render_sidebar():
    with st.sidebar:
        st.header("💬 Cuộc trò chuyện")
        st.divider()

        conn = get_db_connection()
        rows = get_user_conversations(conn, st.session_state.user_id)

        options = [(conv_id, title or "Cuộc trò chuyện chưa có tiêu đề") for conv_id, title in rows]
        current_conv_id = st.session_state.conv_id

        if not options:
            st.caption("Chưa có cuộc trò chuyện nào để tải lại.")
            return

        option_ids = [conv_id for conv_id, _ in options]
        option_title_map = {conv_id: title for conv_id, title in options}

        default_index = 0
        if st.session_state.selected_conversation_id in option_ids:
            default_index = option_ids.index(st.session_state.selected_conversation_id) #lấy index đầu tiên của conversation được chọn
        elif current_conv_id in option_ids:
            default_index = option_ids.index(current_conv_id)


        selected_id = st.selectbox(
            "Chọn cuộc trò chuyện",
            options=option_ids,
            index=default_index,
            format_func=lambda cid: option_title_map[cid],
            key="conversation_selectbox",
        )
        st.session_state.selected_conversation_id = selected_id

        #load ngay lập tức khi người dùng chọn 1 title khác ở selectbox
        if selected_id != current_conv_id:
            load_conversation_into_state(selected_id)
            st.rerun()

render_sidebar()
    
#New chat button to reset conversation, return to home section
st.markdown("""
    <style>
    /* Strategy: inject a marker element before the button.
       Streamlit wraps each st.markdown / st.button in its own
       div.stElementContainer inside a parent stVerticalBlock.
       We use the adjacent-sibling combinator (+) to move from
       the marker's stElementContainer to the button's stElementContainer
       and apply fixed positioning there. */
    div[data-testid="stElementContainer"]:has(#new-chat-marker) + div[data-testid="stElementContainer"] {
        position: fixed !important;
        top: 70px;
        left: 450px;
        z-index: 9999;
    }
    </style>
    """, unsafe_allow_html=True)

def reset_conversation():
    if "messages" in st.session_state:
        st.session_state.messages = []

    #mỗi khi bấm new chat -> sinh ra một id hội thoại mới hoàn toàn
    st.session_state.conv_id = str(uuid.uuid4())
    st.session_state.selected_conversation_id = None
    st.rerun()


# Marker element must be immediately before the button so the CSS sibling selector works
st.markdown('<div id="new-chat-marker"></div>', unsafe_allow_html=True)
if st.button("💬New Chat", key="new-chat-fixed", help="Xóa lịch sử và bắt đầu hội thoại mới"):
    reset_conversation()


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
    


def load_chain(k,temperature):
    return get_chain(k = k, temperature = temperature, embedding_model = embedding_model, _reranker_model = reranker_model)

rag_chain = load_chain(k = RETRIEVER_TOP_K, temperature = LLM_TEMPERATURE) #thay từ điều chỉnh slider -> hardload

#Tach rieng 2 initialization: 

#2 - Day moi la list chua context cua cau hoi cuoi cung (LLM memory)
if "last_context" not in st.session_state:
    st.session_state.last_context = []

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
            conn = get_db_connection()
            title = generate_title(first_q, first_r)
            insert_title_conversations(conn, conv_id, uid, title)


        #thread sinh title độc lập với thread nhập input -> tức người dùng vẫn có thể nhập câu hỏi tiếp theo
        #và thread sinh title chạy ngầm
        fire_and_forget(_generate_and_save_title, session_id, user_id, first_question, first_response)

        
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

@st.dialog("👉Giúp em hiểu tại sao đại ca không thích câu này?")
def feedback_dialog():
    reasons = st.multiselect(
        "Chọn vấn đề đại ca gặp phải:",
        ["Thông tin không chính xác", "Thiếu thông tin", "Thông tin thừa thãi", "Văn phong không phù hợp"],
        key = f"reasons_{msg_len}"
    )

    other_comment = st.text_area("Ghi rõ hơn (nếu có):", key=f"comment_{msg_len}")

    if st.button("Gửi đánh giá chi tiết", key=f"btn_gb_{msg_len}"):
        last_msg = st.session_state.messages[-1]
        last_user_msg = st.session_state.messages[-2] if len(st.session_state.messages) > 1 else {"content": "Unknown"}

        save_feedback(
            last_user_msg["content"],
            last_msg["content"],
            "Dislike",
            reason=", ".join(reasons),
            comment=other_comment
        )
        st.success("Đã ghi nhận, cảm ơn đại ca🙏! Sẽ bảo bot học lại bài thưa đại ca!")

if st.session_state.messages and st.session_state.messages[-1]["role"] == "ai":
    st.write("---")
    st.caption("Đại ca thấy câu trả lời thế nào? (Feedback để giúp em khôn lên)")

    col_fb, col_survey = st.columns([1, 4])

    with col_fb:
        msg_len = len(st.session_state.messages)
        feedback = st.feedback("thumbs", key=f"fb_{msg_len}")

    if feedback == 0:
        feedback_dialog()

    elif feedback == 1:
        last_msg = st.session_state.messages[-1]
        last_user_msg = st.session_state.messages[-2] if len(st.session_state.messages) > 1 else {"content": "Unknown"}
        save_feedback(
            last_user_msg["content"],
            last_msg["content"],
            "Like"
        )
        st.toast(f"Cảm ơn đại ca đã ủng hộ🙏!", icon= "💾")