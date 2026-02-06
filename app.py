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

def load_chain(k,temperature):
    st.write(f"--Đang trả về chain với k ={k} và temperature = {temperature}")
    return get_chain(k = k, temperature = temperature)

rag_chain = load_chain(k = k_value, temperature = temperature_value)
question = st.text_input("Nhập câu hỏi của m đi con")

if question:
    with st.spinner("Đang nghĩ..."):
        session_id = "user_vjp_pro_1"

        response = rag_chain.invoke(
            {"question": question},
            config={"configurable": {"session_id": session_id}}
        )
        answer = response['answer']
        source_documents = response['context']

        st.write("##Câu trả lời:")
        st.write(answer)

        if source_documents:
            st.write("##Các nguồn sử dụng:")
            for i,doc in enumerate(source_documents):
                source = doc.metadata.get("source", "Không rõ nguồn")
                with st.expander(f"Nguồn {i+1}: {source}"):
                    st.write(doc.page_content)


        