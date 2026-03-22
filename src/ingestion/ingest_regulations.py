import os
import shutil
import sys
import pickle
sys.path.append(os.path.abspath('.'))
from langchain.retrievers import ParentDocumentRetriever

from langchain.storage import LocalFileStore, EncoderBackedStore
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.utils import get_embedding_model
from dotenv import load_dotenv

load_dotenv()

CHROMA_PATH = "chroma_db"
DOC_STORE_PATH = "doc_store_pdr"

   
def ingest_regulations():

    if os.path.exists(CHROMA_PATH): shutil.rmtree(CHROMA_PATH)
    if os.path.exists(DOC_STORE_PATH): shutil.rmtree(DOC_STORE_PATH)

    #inject metadata for each doc 
    #để reranker và embeddings có thể hiểu liên hệ giữa page_content và metadata
    #bug(21/3/2026): DOCLING không thể nhận diện layout của 2 bảng có grid gần nhau 
    #in docling: "multiple columns in extracted tables are erroneously merged into one"
    # + nhận diện header cha (#)'Chương' và header con (##)'Điều' cùng cấp
    #first docling through turn off cells matching and accurate base model layout -> failed

    from src.ingestion.splitter import markdown_chunk_documents
    semantic_docs = markdown_chunk_documents


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

    