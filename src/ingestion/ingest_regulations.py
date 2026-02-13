import re
import os
import glob
import shutil
import sys
import pickle
sys.path.append(os.path.abspath('.'))
from langchain_community.document_loaders import PyPDFLoader
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import LocalFileStore, EncoderBackedStore
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from src.utils import get_embedding_model
from dotenv import load_dotenv

load_dotenv()

PDF_PATH = "data_quyche/QCDT_2025_DHBK.pdf"
CHROMA_PATH = "chroma_db"
DOC_STORE_PATH = "doc_store_pdr"

def ingest_regulations():

    if os.path.exists(CHROMA_PATH): shutil.rmtree(CHROMA_PATH)
    if os.path.exists(DOC_STORE_PATH): shutil.rmtree(DOC_STORE_PATH)

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

    parent_docs = []

    #lọc bỏ chunk rỗng
    cleaned_chunks = [chunk.strip() for chunk in raw_chunks if chunk.strip()]

    print(f"Đã cắt được {len(cleaned_chunks)} điều khoản")

    #変換 sang LangChain Documents
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

        parent_docs.append(doc)

    print(f"Đã tạo {len(parent_docs)} Parent Docs (Chunk to - Điều)")

    print("Đang embedding và lưu vào ChromaDB...")
    embedding_model = get_embedding_model()

    
    #Vector store - LƯU CHILD
    vector_store = Chroma(
        collection_name="split_parents",
        embedding_function=embedding_model,
        persist_directory=CHROMA_PATH
    )
    
    #File store (SSD) - LƯU PARENT CHUNK
    fs = LocalFileStore(DOC_STORE_PATH)

    store = EncoderBackedStore(
        store=fs,
        key_encoder=lambda x: x, #key giu nguyen la string
        value_serializer=pickle.dumps, #write: use pickle to convert object documents to bytes
        value_deserializer=pickle.loads #read: from bytes to document object
    )

    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)

    print("Đang Ingest PDR (Cắt Nhỏ -> Embed -> Map ID)")

    retriever = ParentDocumentRetriever(
        vectorstore=vector_store,
        docstore=store,
        child_splitter=child_splitter,
    )

    #nạp parent docs vào
    #Tự động: cắt ra con -> embed con -> lưu cha -> Map id

    """problem: docstore (LocalFileStore) wont receive a document data type, only receive if object is byte type""" 
    retriever.add_documents(parent_docs, ids=None)

    print("Done! PDR DB đã sẵn sàng")


if __name__ == "__main__":
    ingest_regulations()

    