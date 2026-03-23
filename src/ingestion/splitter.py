import uuid

from pathlib import Path
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
_REPO_ROOT = Path(__file__).resolve().parents[2]
MARKDOWN_PATH = _REPO_ROOT / "data_quyche" / "QCDT_2025_DHBK.md"

markdown_documents: str = MARKDOWN_PATH.read_text(encoding="utf-8")

headers_to_split_on = [
    ("#", "Chương"),
    ("###", "Điều"),
]

markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

markdown_chunk_documents = markdown_splitter.split_text(markdown_documents)

parent_splitter = RecursiveCharacterTextSplitter(chunk_size = 2000, chunk_overlap = 200)
    #2000 chunk_size approx 6000 char -> approx 3000 token, tránh vượt context window

child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)


def get_pdr_data(vector_store, store):
    for doc in markdown_chunk_documents:
        chuong = doc.metadata.get('Chương', '')
        dieu = doc.metadata.get('Điều', '')

        #Tạo metadata 'title' to sync with qa_chain
        doc.metadata['title'] = f"{chuong} - {dieu}"

        #chia thành các parent lớn
        parents = parent_splitter.split_documents([doc])

        for p_doc in parents:
            #create distinct id for each parent
            parent_id = str(uuid.uuid4())

            #chia parent thành các child
            children = child_splitter.split_documents([p_doc])

            #inject metadata for child
            context_prefix = f"Ngữ cảnh: {p_doc.metadata['title']}\nNội dung: "

            for c_doc in children:
                c_doc.page_content = context_prefix + c_doc.page_content
                #gán id để link về parent
                c_doc.metadata["doc_id"] = parent_id

            vector_store.add_documents(children)
            store.mset([(parent_id, p_doc)])

