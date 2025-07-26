import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
load_dotenv()

chroma_path = "chroma_db"

def store_embeddings(chunks):
    try:
        embedding_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        print("Đang lưu vector...")

        vector_store = Chroma.from_documents(
            documents = chunks, 
            embedding = embedding_model,
            persist_directory= chroma_path
        #khi không chỉ định directory, chroma chỉ tự động lưu vào ram. Khi .py script kết thúc, data bị all wiping
        )
        print(f"Lưu thành công vào thư mục {chroma_path}")
        return vector_store
    except Exception as e:
        print(f"Lỗi khi tạo vector store: {str(e)}")
        raise
    

