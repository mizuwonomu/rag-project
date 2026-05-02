import streamlit as st
from src.database.conversation_queries import get_conversation_messages

def load_conversation_into_state(conversation_id: str, db_connection_factory):
    """Load lại message từ database, chia rõ type message để render"""
    conn = db_connection_factory()
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