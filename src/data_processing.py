import re

file_path = "D:\\rag-project\\data\\summary.txt"
def load_and_clean_content(file_path):
    with open(file_path, 'r', encoding = "utf-8") as f:
        text = f.read()
    
    #Xóa khoảng trắng thừa ở đầu và cuối dòng
    text = "\n".join(line.strip() for line in text.splitlines())

    #Thay thế nhiều dấu xuống dòng bằng 1 dấu 
    text = re.sub(r'\n{2,}', 'n', text)

    print("Đã xong")
    return text
def chunk_text(text, chunk_size):
    print(f"Cắt text thành các chunk độ dài {chunk_size} kí tự")
    chunks = []

    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size] #cắt từ vị trí i đến vị trí i + chunk_size
        #Nếu đoạn cuối có độ dài ngắn hơn chunk_size, sẽ tự động lấy hết văn bản
        chunks.append(chunk)
    
    return chunks


