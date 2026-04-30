import streamlit as st


def render_new_chat_button(on_reset) -> None:
    st.markdown("""
        <style>
        div[data-testid="stElementContainer"]:has(#new-chat-marker) + div[data-testid="stElementContainer"] {
            position: fixed !important;
            top: 70px;
            left: 450px;
            z-index: 9999;
        }
        </style>
        """, unsafe_allow_html=True)

    st.markdown('<div id="new-chat-marker"></div>', unsafe_allow_html=True)
    if st.button("💬New Chat", key="new-chat-fixed", help="Xóa lịch sử và bắt đầu hội thoại mới"):
        on_reset()
        st.rerun()
