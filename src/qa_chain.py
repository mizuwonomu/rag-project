import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain_chroma import Chroma
from langchain.schema.runnable import RunnablePassthrough #chạy nhiều nhánh xử lý cùng 1 lúc
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableLambda
from dotenv import load_dotenv

load_dotenv()
CHROMA_PATH = "chroma_db"

def get_chain(k, temperature):
    embedding_model = GoogleGenerativeAIEmbeddings(model = "models/gemini-embedding-001")
    #load vector store
    vector_store = Chroma(
        embedding_function = embedding_model,
        persist_directory = CHROMA_PATH
    )
    llm = GoogleGenerativeAI(model = "models/gemini-2.5-flash", temperature = temperature)

    def classifier(question: str):

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
        search_type = "mmr", #Dùng maximal marginal relevance
        search_kwargs = {
            'k': k, 
            'fetch_k': 20,
            'filter': dynamic_filter
            }
        )
        return retriever.invoke(question)

    #define template và prompt
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
            context = RunnableLambda(retrieve_with_filter),
            #chuyển đổi object retreive thành 1 Runnable
            #để có thể delay cho đến khi được invoke đúng thời điểm, thay vì chạy ngay tức khắc
            question = RunnablePassthrough(),
    ).assign(
        #answer được chạy qua 1 chain con
        answer = answer_chain
    )

    return rag_chain