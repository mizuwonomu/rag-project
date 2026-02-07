import re
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

PDF_PATH = "data_quyche/QCDT_2025_DHBK.pdf"
CHROMA_PATH = "chroma_db"

def ingest_regulations():
    #1.読み込み
    print(f"Đang load file: {PDF_PATH}...")
    loader = PyPDFLoader(PDF_PATH)
    raw_pages = loader.load()

    #2. Nối tất cả các trang thành 1 string lớn
    #để rồi xử lý regex
    full_text = "\n".join([page.page_content for page in raw_pages])

    #3. Tách theo pattern "Điều"
    #"Điều" + số + chấm. Ví dụ là: "Điều 1.", "Điều 2."
    pattern = r"(?=Điều \d+\.)"

    raw_chunks = re.split(pattern, full_text)

    #lọc bỏ chunk rỗng
    cleaned_chunks = [chunk.strip() for chunk in raw_chunks if chunk.strip()]

    print(f"Đã cắt được {len(cleaned_chunks)} điều khoản")

    #変換 sang LangChain Documents
    documents = []
    for chunk in cleaned_chunks:
        #trích xuất tiêu đề của từng điều khoản
        lines = chunk.split('\n')
        title = lines[0] if lines else "Unknown Article"

        doc = Document(
            page_content=chunk,
            metadata={
                "source": "Quy chế đào tạo 2025",
                "title": title,
                "category": "regulation"
            }
        )

        documents.append(doc)

    print("Đang embedding và lưu vào ChromaDB...")
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    #delete old DB
    import shutil
    if os.path.exists(CHROMA_PATH): shutil.rmtree(CHROMA_PATH)

    Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=CHROMA_PATH
    )
    print("DB đã sẵn sàng.")

if __name__ == "__main__":
    ingest_regulations()

    