import os
import sys
sys.path.append(os.path.abspath('.'))
import pickle
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.storage import LocalFileStore, EncoderBackedStore
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.schema.runnable import RunnablePassthrough #chạy nhiều nhánh xử lý cùng 1 lúc
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableLambda
from langsmith import traceable

from dotenv import load_dotenv
load_dotenv()


CHROMA_PATH = "chroma_db"
DOC_STORE_PATH = "doc_store_pdr"

def format_docs(docs):
    formatted = []
    #Van la ep string tu document object, nhung co them cac metadata de danh dau
    
    for i, doc in enumerate(docs):
        source_title = doc.metadata.get("title", f"Điều khoản {i+1}")
        content = doc.page_content.replace("\n", " ")
        formatted.append(f"Source [{i+1}] ({source_title}):\n{content}")

    return "\n\n".join(formatted)

store = {}
def get_session_history(session_id : str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    
    return store[session_id]


@traceable(run_type='chain')
def get_chain(k, temperature, embedding_model, reranker_model):
    embedding_model = embedding_model
    #load vector store
    vector_store = Chroma(
        collection_name= "split_parents",
        embedding_function = embedding_model,
        persist_directory = CHROMA_PATH
    )

    fs = LocalFileStore(DOC_STORE_PATH)
    doc_store = EncoderBackedStore(
        store=fs,
        key_encoder=lambda x: x,
        value_serializer=pickle.dumps,
        value_deserializer=pickle.loads
    )

    child_vector_retriever = vector_store.as_retriever(
        search_type = "similarity",
        search_kwargs = {'k': k}
    )

    child_data = vector_store.get()

    from langchain_core.documents import Document

    all_child_docs = [
        Document(page_content=txt, metadata=md)
        for txt, md in zip(child_data['documents'], child_data['metadatas'])
    ]

    bm25_retriever = BM25Retriever.from_documents(all_child_docs)
    bm25_retriever.k = k

    ensemble_retriever = EnsembleRetriever(
        retrievers = [child_vector_retriever, bm25_retriever],
        weights = [0.5, 0.5]
    )

    #test hybrid with no filter
    llm = ChatGroq(
        model = "qwen/qwen3-32b",
        temperature = temperature,
        reasoning_format = "parsed"
    )

    #hugging-face based reranker (bge-reranker-v2-m3 by default)
    reranker = reranker_model


    #Custom chain de lay parent:
    #Query -> Ensemble -> List[child] -> rerank -> extract ids -> docstore -> list[parent]
    def retreive_parents(input_data):
        #find child
        question = input_data["standalone_question"]

        candidate_child_docs = ensemble_retriever.invoke(question)

        if not candidate_child_docs:
            return []

        #limit to top 15 for reranking (independent of k)
        candidate_child_docs = candidate_child_docs[:15]

        pairs = [(question, doc.page_content) for doc in candidate_child_docs]
        scores = reranker.predict(pairs)

        #zip docs with scores and sort descending

        scored_docs = list(zip(candidate_child_docs, scores))
        scored_docs.sort(key=lambda x: float(x[1]), reverse=True)

        for doc, score in scored_docs:
            try:
                doc.metadata["rerank_score"] = float(score)

            except Exception:
                pass

        # all children that pass the primary / fallback thresholds
        thresholded_children = [doc for doc, s in scored_docs if float(s) >= 0.8]

        #fallback threshold >= 0.6 if nothing passes 0.8
        if not thresholded_children:
            thresholded_children = [doc for doc, s in scored_docs if float(s) >= 0.6]

        if not thresholded_children:
            return []

        seen_parent_ids = set()
        unique_parent_ids = []

        for doc in thresholded_children:
            p_id = doc.metadata.get("doc_id")

            if p_id is not None and p_id not in seen_parent_ids:
                seen_parent_ids.add(p_id)
                unique_parent_ids.append(p_id)

        if not unique_parent_ids:
            return []

        parent_docs_raw = doc_store.mget(unique_parent_ids)
        parents = [p for p in parent_docs_raw if p is not None]

        max_parents = 3
        if len(parents) > max_parents:
            parents = parents[:max_parents]

        return parents
    rephrase_system_prompt = """"Cho trước lịch sử trò chuyện và câu hỏi mới nhất của người dùng (câu hỏi này có thể chứa các thông tin tham chiếu đến ngữ cảnh trong lịch sử trò chuyện), 
    hãy tạo ra một câu hỏi độc lập có thể hiểu được mà không cần đến lịch sử trò chuyện. 
    KHÔNG trả lời câu hỏi, chỉ viết lại nó nếu cần thiết, nếu không thì trả về nguyên bản."
    Chỉ được trả lời bằng TIẾNG VIỆT.
    """

    rephrase_prompt = ChatPromptTemplate.from_messages([
        ("system", rephrase_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])

    #chain nhỏ chỉ làm nhiệm vụ: Input (History + Query) -> Output (String query mới)
    rephrase_chain = rephrase_prompt | llm | StrOutputParser()

    #Phân loại câu hỏi thuộc rag hay xã giao
    router_template = """
    M là một chuyên gia phân loại câu hỏi. Hãy đọc câu hỏi của người dùng và quyết định câu hỏi đó thuộc loại nào"

    1. 'chat': Các câu chào hỏi xã giao, hỏi thăm sức khỏe, không liên quan đến thông tin cụ thể trong tài liệu (VD: Chào bạn, bạn tên gì?, ...)
    2. 'RAG': 
        - Các câu hỏi liên quan đến quy chế đào tạo, Luật, Quy định nhà trường
        - Từ khóa nhận diện: Ví dụ: "học phí", "tín chỉ", "cảnh báo học tập", "điểm học phần",...
        - Các câu hỏi về thủ tục, điều kiện, thời gian học tập tại HUST.

    Chỉ được trả kết quả theo 1 từ duy nhất: 'chat' hoặc 'RAG'. KHÔNG trả lời thêm bất cứ điều gì khác.

    Câu hỏi: {question}
    Phân loại:
    """


    router_prompt = ChatPromptTemplate.from_template(router_template)

    router_chain = router_prompt | llm | StrOutputParser()
    
    #Nhánh A: Chat thông thường(dùng cách xã giao của model)
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", "M là một trợ lý AI vui tính. Hãy trò chuyện thân thiện với người dùng. Gọi user là 'mày', bản thân là 'tao', có thể viết tắt thành 't' và 'm'."),

        MessagesPlaceholder(variable_name="chat_history"),
        
        ("human", "{question}"),
    ])
    
    #Chuẩn hóa output format giống rag chain, trả về dạng 辞書型
    chat_chain = (
        chat_prompt 
        | llm 
        | StrOutputParser()
        | RunnableLambda(lambda x: {"answer": x, "context": []}) #fake content rỗng
    )

    #define prompt with role-based messages, only rely on documents
    qa_prompt = ChatPromptTemplate.from_messages([

        ("system", """M là trợ lý AI chuyên hỗ trợ sinh viên HUST - Đại học Bách Khoa tra cứu Quy chế đào tạo (Academic Regulations)
        Nhiệm vụ của m là trả lời chính xác dựa trên Context được cung cấp.
        
        Quy tắc tuyệt đối (絶対ルール):
        1. Luôn trích dẫn rõ ràng thông tin nằm ở **Điều nào** (Dựa vào dòng đầu tiên của context). Với các con số (tín chỉ, học phí, mức cảnh báo), phải tuyệt đối chính xác.
        2. KHÔNG được sử dụng kiến thức bên ngoài (Outside knowledge) hoặc kiến thức có sẵn trong model (Pre-trained knowledge) để trả lời.
        3. Nếu thông tin không có trong Context, hãy trả lời: "Thông tin này không có trong quy chế hiện tại."
        4. KHÔNG được bịa đặt nội dung (Hallucination). 
        5. Giọng văn nghiêm túc nhưng dễ hiểu, phù hợp với sinh viên.

        Quy tắc trích dẫn (Citation Rules): 
        1. Khi đưa ra bất kỳ thông tin nào, PHẢI trích dẫn nguồn gốc bằng cú pháp [index]. Ví dụ: "Theo quy định về học phí [1], sinh viên phải..." 
        2. Nếu thông tin đến từ nhiều nguồn, hãy liệt kê đủ: [1], [3]. 
        3. Cuối câu trả lời KHÔNG cần tạo danh sách tài liệu tham khảo (vì giao diện sẽ tự hiển thị).   

        Context:
        {context}
        """),

        MessagesPlaceholder(variable_name="chat_history"),

        ("human", "{question}"),
    ])
    
    answer_chain = (
        qa_prompt
        | llm
        | StrOutputParser()
    )
    rag_chain=RunnableParallel(
        #Nhánh 1: tính toán câu hỏi độc lập

        input_pass = RunnablePassthrough(), #giữ input gốc để lấy chat_history
        #passthrough truyền nguyên 1 dict {'question', 'chat_history}

        standalone_question = rephrase_chain, #dùng llm để viết lại câu hỏi

    ).assign(
        #nhánh 2: dùng câu hỏi độc lập để tìm tài liệu (context)

        context = RunnableLambda(retreive_parents)
    
        #x lúc này đã có key 'standalone_question'
        #lambda để chuyển đổi object retreive thành 1 Runnable
        #để có thể delay cho đến khi được invoke đúng thời điểm, thay vì chạy ngay tức khắc

    ).assign( #Lấy context là danh sách các chunk gốc và question
        #nhánh 3: Trả lòi
        answer = (
            RunnableLambda(lambda x: { #x: chứa dictionary của output runnableparallel 
                "context": format_docs(x["context"]),
                "question": x["input_pass"]["question"],
                "chat_history": x["input_pass"]["chat_history"]
            }) #ép danh sách chunk thành string
            | answer_chain
        )
        #Đảm bảo app.py nhận 1 cục dictionary đầy đủ
    )


    def route_decision(info):
        #info: Dict chứa đầu ra của các bước trước

        #router_chain cần input là dict {'question':...}
        decision = router_chain.invoke({"question": info["question"]})

        #clean string (xóa khoảng trắng thừa nếu có)
        decision = decision.strip().lower()

        if "chat" in decision:
            return chat_chain
        else:
            return rag_chain
    
    # Tổng hợp final chain
    full_chain = RunnableLambda(route_decision)

    chain_with_history = RunnableWithMessageHistory(
        full_chain,
        get_session_history,
        input_messages_key= "question",
        history_messages_key= "chat_history",
        output_messages_key= "answer",
    )
    
    return chain_with_history

def debug_memory(session_id):

    if session_id not in store:
        return ["Chưa có lịch sử chat nào trong RAM này!"]

    history_obj = store[session_id]

    readable_history = []

    for msg in history_obj.messages:
        readable_history.append({
            "Role": msg.type.upper(), #msg type could be ai or human role
            "Content": msg.content 
        })

    return readable_history