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

def chunking_doc(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 50, #dùng để  "nhắc lại các số lượng kí tự cuối". ví dụ như chunk 1 có 50 kí tự cuối, thì mở đầu chunk 2 cx sẽ có bấy nhiêu kí tự từ chunk 1
        length_function = len, #sử dụng chính hàm len để đo độ dài chuỗi, đánh dấu toàn bộ như dấu cách, kí tự,..
    )

    docs_to_split = [Document(page_content = text)] #Convert string sang object, 1 list chứa 1 document
    chunks = text_splitter.split_documents(docs_to_split) #nhận vào list document và trả về list document
    print(f"Cắt thành công {len(chunks)} chunks.")

    return chunks

