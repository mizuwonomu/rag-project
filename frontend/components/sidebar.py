import streamlit as st
from src.database.conversation_queries import get_user_conversations
from frontend.services.conversation_loader import load_conversation_into_state

def _on_conversation_selected():
    #interaction của chính user, thực hiện 1 lần trong render sidebar
    st.session_state.load_selected_conversation = True

def render_sidebar(db_connection_factory):
    with st.sidebar:
        st.header("💬 Cuộc trò chuyện")
        st.divider()

        conn = db_connection_factory()
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

        if st.session_state.conversation_selectbox_id not in option_ids:
            st.session_state.conversation_selectbox_id = option_ids[default_index]

        selected_id = st.selectbox(
            "Chọn cuộc trò chuyện",
            options=option_ids,
            index=default_index,
            format_func=lambda cid: option_title_map[cid],
            key="conversation_selectbox_id",
            on_change=_on_conversation_selected
        )
        st.session_state.selected_conversation_id = selected_id

        #chỉ load khi người dùng thay đổi selectbox value, tức chọn conversation khác nhau trong selectbox
        if st.session_state.load_selected_conversation and selected_id != current_conv_id:
            st.session_state.load_selected_conversation = False
            load_conversation_into_state(selected_id, db_connection_factory)
            st.rerun()
        st.session_state.load_selected_conversation = False #reset flag nếu có gặp new chat trigger -> không còn query và tiếp tục render message cũ 
                                                            #của cuộc hội thoại gần nhất