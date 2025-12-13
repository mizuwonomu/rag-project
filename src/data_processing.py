import re
from markdown import markdown
from langchain_text_splitters import RecursiveCharacterTextSplitter
from bs4 import BeautifulSoup
from langchain_core.documents import Document
def doc_cleaning(file_path):
    print(f"Đang đọc file {file_path}, bình tĩnh")
    with open(file_path, "r", encoding= "utf-8") as f:
        text = f.read()
    return text

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

    parts = re.split(r'(?=\n## [IVX]+\.)', text)

    #?=: Regex chỉ định nội dung trước đó, thay vì cắt tại chính nội dung mang nội dung trên, regex sẽ tự động cắt kí tự "ngay trước" nội dung chỉ định
    #Ví dụ: Nội dung A,B,C \n II. Phân tích tâm lí -> Cắt ra thành 2 list section, 1 là  [A,b,c] 2 là [II. Phân tích tâm lí] thay vì cắt luôn nội dung "II."
    
    #\n: Vì mỗi mục đều luôn có dấu xuống dòng trước section đó
    #\. để chỉ định đây là dấu chấm, không phải kí tự đặc biệt
    #|: Toán tử or
    #[]: Character set: Chỉ định rằng kí tự tiếp theo phải là MỘT
    #trong các kí tự nằm bên trong dấu ngoặc vuông
    #IVX: Các kí tự được phép: Chỉ các chữ cái I, V hoặc X (1,5,10)
    #+: Bộ định lượng, chỉ định rằng kí tự đứng trước nó phải xuất hiện 1 lần trở lên
    #Ví dụ: I,II, III (1,2,3 lần kí tự I)
    
    for part in parts:
        if not part.strip():
            continue

        part_striped = part.strip()
        section_name = "Unknown"
        if (part_striped.startswith("## I. DIỄN BIẾN")): #update: no more \n regex, do strip đã xóa 
            section_name = "Diễn biến"

        elif (part_striped.startswith("## II. PHÂN TÍCH TÂM LÝ")):
            section_name = "Phân tích tâm lý"

        elif (part_striped.startswith("## III. Ý NGHĨA")):
            section_name = "Ý nghĩa"

        elif (part_striped.startswith("## IV. KẾT LUẬN")):
            section_name = "Kết luận"


        html = markdown(part)
        soup = BeautifulSoup(html, "html.parser")
        cleaned_text = soup.get_text()
        cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text).strip()


        section_metadata = base_metadata.copy()
        section_metadata.update({
            "section": section_name
        }
        )

        #Tạo 1 document riêng cho mỗi phần nội dung section. 
        #Khi này, sẽ thành 4 document có list 4 phần section nội dung riêng
        doc = Document(page_content= cleaned_text, metadata = section_metadata)
        docs_to_split.append(doc)

        #Khi này, docs_to_split đang có 4 list riêng
            #docs_to_split = [
            #Document(page_content = "Nội dung diễn biến...", metadata = "Diễn biến")
            #Document(page_content = "Nội dung phân tích...", metadata = "Phân tích")
            #.... Tiếp tục 

    chunks = text_splitter.split_documents(docs_to_split) 
    #khi này, text splitter sẽ nhận biết từng section để cắt cho đến cuối dãy, hay hết 1 content section

    
    for i,chunk in enumerate(chunks):
        chunk.metadata.update({
            "chunk_id": i, 
            "chunk_index": f"{i+1}/{len(chunks)}", #chỉ định vị trí index thứ mấy trong tổng chunk, ví dụ như chunk 3/8
            "chunk_size": len(chunk.page_content), #độ dài thực tế của chunk 
            "chunk_start_char": i * max(0,chunk_size - chunk_overlap) #"highlight" đúng đoạn văn bản gốc trên giao diện khi trích dẫn nguồn
        })
    for chunk in chunks:
        chunk.metadata["total_chunks"] = len(chunks)
    print(f"Cắt thành công {len(chunks)} chunks.")

    return chunks

