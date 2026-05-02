import streamlit as st
from frontend.components.source_panel import render_sources

def stream_handler(chain, question: str, session_id: str):

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

def render_streamed_ai_answer(chain, question: str, session_id: str):
    with st.chat_message("ai"):
        full_response = st.write_stream(stream_handler(chain, question, session_id))
        sources = st.session_state.get("last_context", [])
        render_sources(sources)
    return full_response, sources