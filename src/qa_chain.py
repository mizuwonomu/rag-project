import os
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.schema.runnable import RunnablePassthrough #chạy nhiều nhánh xử lý cùng 1 lúc
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableLambda

from dotenv import load_dotenv
load_dotenv()


CHROMA_PATH = "chroma_db"

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

store = {}
def get_session_history(session_id : str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    
    return store[session_id]


def get_chain(k, temperature):
    embedding_model = GoogleGenerativeAIEmbeddings(model = "models/gemini-embedding-001")
    #load vector store
    vector_store = Chroma(
        embedding_function = embedding_model,
        persist_directory = CHROMA_PATH
    )

    chroma_retriever = vector_store.as_retriever(
        search_type = "similarity",
        search_kwargs = {'k': k}
    )

    data = vector_store.get()

    from langchain_core.documents import Document

    docs_for_bm25 = [
        Document(page_content=txt, metadata=md)
        for txt, md in zip(data['documents'], data['metadatas'])
    ]
    bm25_retriever = BM25Retriever.from_documents(docs_for_bm25)
    bm25_retriever.k = k

    ensemble_retriever = EnsembleRetriever(
        retrievers = [chroma_retriever, bm25_retriever],
        weights = [0.5, 0.5]
    )

    #test hybrid with no filter
    llm = GoogleGenerativeAI(model = "models/gemini-3-flash-preview", temperature = temperature)

    '''def classifier(question: str):

        template = """
        Câu hỏi của người dùng là: {question}
        Pphân loại nó thuộc section nào dưới đây:

        1. Diễn biến
        2. Phân tích tâm lý
        3. Ý nghĩa
        4. Tổng kết
        5. Không rõ

        Chỉ được chọn 1 trong 5 section trên. 
        """

        prompt = template.format(question = question)
        section = llm.invoke(prompt)

        if ("Diễn biến" in section):
            return {"section": "Diễn biến"}

        elif ("Phân tích" in section):
            return {"section": "Phân tích tâm lý"}
        
        elif ("Ý nghĩa" in section):
            return {"section": "Ý nghĩa"}
        
        elif ("Kết luận" in section):
            return {"section": "Kết luận"}

        else:
            return {} #Seciton rỗng, aka không thuộc section nào
    
    #Generate retriever
    def retrieve_with_filter(question: str):
        dynamic_filter = classifier(question)
        retriever = vector_store.as_retriever(
        search_type = "similarity", #Dùng maximal marginal relevance
        search_kwargs = {
            'k': k, 
            'filter': dynamic_filter
            }
        )
        return retriever.invoke(question)
    '''

    def retrieve_hybrid(input_data):
        question = input_data["standalone_question"]

        docs = ensemble_retriever.invoke(question)

        return docs[:k]

    rephrase_system_prompt = """Given a chat history and the lastest user quesion
    which might reference context in the chat history, formulate a standalone question which can be
    understood without the chat history. Do NOT answer the question, just reformulate it if needed 
    and otherwise return it as is.
    """

    rephrase_prompt = ChatPromptTemplate.from_messages([
        ("system", rephrase_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])

    #chain nhỏ chỉ làm nhiệm vụ: Input (History + Query) -> Output (String query mới)

    rephrase_chain = rephrase_prompt | llm | StrOutputParser()
    #define prompt with role-based messages

    qa_prompt = ChatPromptTemplate.from_messages([

        ("system", """M là trợ lý AI chuyên về trả lời nội dung cho phim Interstellar - bộ phim khoa học viễn tưởng về vũ trụ.
        Dựa vào context dưới đây để trả lời. Nếu không biết, hãy trả lời 'T không có câu trả lời cho câu hỏi trên'.
        
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

        context = RunnableLambda(retrieve_hybrid)
    
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

    chain_with_history = RunnableWithMessageHistory(
        rag_chain,
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