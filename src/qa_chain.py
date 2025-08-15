import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain_chroma import Chroma
from langchain.schema.runnable import RunnablePassthrough #chạy nhiều nhánh xử lý cùng 1 lúc
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel
from dotenv import load_dotenv

load_dotenv()
CHROMA_PATH = "chroma_db"

def get_chain(k = 3, temperature = 0.4):
    embedding_model = GoogleGenerativeAIEmbeddings(model = "models/gemini-embedding-001")
    #load vector store
    vector_store = Chroma(
        embedding_function = embedding_model,
        persist_directory = CHROMA_PATH
    )
    #Generate retriever
    retriever = vector_store.as_retriever(search_kwargs = {'k': k}) #tìm 3 chunks liên quan nhất

    llm = GoogleGenerativeAI(model = "models/gemini-2.5-flash", temperature = temperature)

    template = """
    Dựa vào những thông tin dưới đây để trả lời câu hỏi. Nếu không thể tìm thấy câu trả lời cho câu hỏi, hãy trả lời 'Tôi không thể tìm thấy
    câu trả lời cho câu hỏi trên'. Tuyệt đối không được bịa câu trả lời.
    Context:
    {context}
    Question:
    {question}

    Answer:
    """
    prompt = ChatPromptTemplate.from_template(template)
    answer_chain = (
        prompt
        | llm
        | StrOutputParser()
    )
    rag_chain=RunnableParallel(
            context = retriever,
            question = RunnablePassthrough(),
    ).assign(
        #answer được chạy qua 1 chain con
        answer = answer_chain
    )

    return rag_chain