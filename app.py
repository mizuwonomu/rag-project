import asyncio
try:
    asyncio.get_event_loop() #Check if event loop already exists
except RuntimeError: #phòng trường hợp event tạo ở 1 main thread khác
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
import streamlit as st
from src.qa_chain import get_chain, debug_memory

st.title("🤖 Hỏi đáp Interstellar")
#sidebar điều chỉnh kawrgs, temp
with st.sidebar:
    st.header("⚙️ Tùy chỉnh tham số")

    k_slider = st.slider(
        "Số lượng chunk tìm kiếm: (k):",
        min_value = 1,
        max_value = 10,
        value = 3,
        step = 1,
    )
    temperature_slider = st.slider(
        "Temperature:",
        min_value = 0.0,
        max_value = 1.0,
        value = 0.4,
        step = 0.1,
    )

    st.info("Đây là những gì bot đang nhớ hiện tại")

    current_session_id = "user_vjp_pro_1"

    memory_content = debug_memory(current_session_id)
    st.json(memory_content)

    if st.button("🗑️ Xóa Trí Nhớ (Clear RAM)"):
        from src.qa_chain import store
        if current_session_id in store:
            del store[current_session_id]
            st.rerun()
    

k_value = k_slider
temperature_value = temperature_slider

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
    st.write(f"--Đang trả về chain với k ={k} và temperature = {temperature}")
    return get_chain(k = k, temperature = temperature)

rag_chain = load_chain(k = k_value, temperature = temperature_value)
question = st.text_input("Nhập câu hỏi của m đi con")

if "last_context" not in st.session_state:
    st.session_state.last_context = []

if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("ai"):

        session_id = "user_vjp_pro_1"

        #dùng st.write_stream để nhận generator 'yield' ở trên
        full_response = st.write_stream(stream_handler(rag_chain, question, session_id))

        if st.session_state.last_context:
            with st.expander("📚 Nguồn tài liệu (Context)"):
                st.write(st.session_state.last_context)




        