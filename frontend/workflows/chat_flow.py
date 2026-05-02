import streamlit as st
from src.database.conversation_queries import insert_title_conversations
from workflows.chat_stream import render_streamed_ai_answer

def handle_query(question: str, deps):
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
            conn = deps.db_connection_factory()
            title = deps.title_generator(first_q, first_r)
            insert_title_conversations(conn, conv_id, uid, title)


        #thread sinh title độc lập với thread nhập input -> tức người dùng vẫn có thể nhập câu hỏi tiếp theo
        #và thread sinh title chạy ngầm
        deps.background_scheduler(_generate_and_save_title, session_id, user_id, first_question, first_response)
        st.session_state.pending_sidebar_title_sync = True

    should_refresh_sidebar_title = (
        len(st.session_state.messages) == 3 and st.session_state.pending_sidebar_title_sync
    )

    if should_refresh_sidebar_title:
        st.session_state.pending_sidebar_title_sync = False
        st.rerun()