import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from dotenv import load_dotenv

load_dotenv()
CHROMA_PATH = "chroma_db"

def get_chain():
    embedding_model = GoogleGenerativeAIEmbeddings(model = "models/gemini-embedding-001")
    #load vector store
    vector_store = Chroma(
        embedding_function = embedding_model,
        persist_directory = CHROMA_PATH
    )

    #Generate retriever
    retriever = vector_store.as_retriever(search_kwargs = {'k': 3}) #tìm 3 chunks liên quan nhất

    llm = GoogleGenerativeAI(model = "models/gemini-2.5-flash", temperature = 0.4)

    template = """
    Dựa vào những thông tin dưới đây để trả lời câu hỏi. Nếu không thể tìm thấy câu trả lời cho câu hỏi, hãy trả lời 'Tôi không thể tìm thấy
    câu trả lời cho câu hỏi trên'. Tuyệt đối không được bịa câu trả lời.
    Context:
    {context}
    Question:
    {input}

    Answer:
    """
    prompt = ChatPromptTemplate.from_template(template)

    rag_chain = (
        {"context":retriever, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser() #lấy kết quả từ gemini và chuyển thành text, chọn string có top result cao nhất
    )

    return rag_chain


