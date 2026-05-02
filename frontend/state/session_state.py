import uuid
import streamlit as st


def bootstrap_session_state() -> None:
    defaults = {
        "user_id": "user_vjp_pro_1",
        "conv_id": str(uuid.uuid4()),
        "messages": [],
        "title_generation_started": set(),
        "selected_conversation_id": None,
        "conversation_selectbox_id": None,
        "load_selected_conversation": False,
        "last_context": [],
        "pending_sidebar_title_sync": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_conversation_state() -> None:
    st.session_state.messages = []
    st.session_state.conv_id = str(uuid.uuid4())
    st.session_state.selected_conversation_id = None
    st.session_state.conversation_selectbox_id = None
    st.session_state.load_selected_conversation = False
    st.session_state.pending_sidebar_title_sync = False
