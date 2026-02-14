import asyncio
try:
    asyncio.get_event_loop() #Check if event loop already exists
except RuntimeError: #phòng trường hợp event tạo ở 1 main thread khác
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
import streamlit as st
from src.qa_chain import get_chain, debug_memory
from src.utils import get_embedding_model

embedding_model = get_embedding_model()

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
        value = 0.1,
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
    return get_chain(k = k, temperature = temperature, embedding_model = embedding_model)

rag_chain = load_chain(k = k_value, temperature = temperature_value)

#Tach rieng 2 initialization: 

#2 - Day moi la list chua context cua cau hoi cuoi cung (LLM memory)
if "last_context" not in st.session_state:
    st.session_state.last_context = []

def handle_query(question):
    with st.chat_message("user"):
        st.markdown(question)

    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("ai"):

        session_id = "user_vjp_pro_1"

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
#Init state
if "messages" not in st.session_state:
    st.session_state.messages = []

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
if not st.session_state.messages:
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
        "Điều kiện nhận học bổng KKHT",
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