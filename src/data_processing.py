import re
from markdown import markdown
from langchain_text_splitters import RecursiveCharacterTextSplitter
from bs4 import BeautifulSoup
from langchain_core.documents import Document
def doc_cleaning(file_path):
    print(f"Đang đọc file {file_path}, bình tĩnh")
    with open(file_path, "r", encoding= "utf-8") as f:
        text = f.read()
    
    #Convert từ markdown sang html
    html = markdown(text)

    #Lột các tag html, chỉ lấy text

    soup = BeautifulSoup(html, "html.parser")
    cleaned_text = soup.get_text()

    #Cleaning lần cuối
    cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text).strip()
    #bất cứ đoạn nào có nhiều hơn enter 3 lần liên tiếp, sẽ chuyển về định dạng ngắt dòng chuẩn là \n\n
    return cleaned_text

def chunking_doc(text, file_path = None, additional_metadata= None, chunk_size = 500, chunk_overlap = 50):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap, #dùng để  "nhắc lại các số lượng kí tự cuối". ví dụ như chunk 1 có 50 kí tự cuối, thì mở đầu chunk 2 cx sẽ có bấy nhiêu kí tự từ chunk 1
        length_function = len, #sử dụng chính hàm len để đo độ dài chuỗi, đánh dấu toàn bộ như dấu cách, kí tự,..
    )
    from datetime import datetime
    base_metadata = { #mẫu chung để định danh cho các chunks
        "source": file_path or "unknown",
        "source_type": "txt",
        "processed_date": datetime.now().isoformat(),
        "original_length": len(text),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "version": "1.0" 
    #Đây là các dictionary 
    }

    if additional_metadata:
        base_metadata.update(additional_metadata) #nếu như có bất cứ thông tin định dạng nào cho vào, update thêm

    docs_to_split = [Document(page_content = text, metadata = base_metadata)] #Convert string sang object, 1 list chứa 1 document
    chunks = text_splitter.split_documents(docs_to_split) #nhận vào list document và trả về list document
    
    for i,chunk in enumerate(chunks):
        chunk.metadata.update({
            "chunk_id": i, 
            "chunk_index": f"{(i+1)/len(chunks)}", #chỉ định vị trí index thứ mấy trong tổng chunk, ví dụ như chunk 3/8
            "chunk_size": len(chunk.page_content), #độ dài thực tế của chunk 
            "chunk_start_char": i * max(0,chunk_size - chunk_overlap) #"highlight" đúng đoạn văn bản gốc trên giao diện khi trích dẫn nguồn
        })
    for chunk in chunks:
        chunk.metadata["total_chunks"] = len(chunks)
    print(f"Cắt thành công {len(chunks)} chunks.")

    return chunks

