import glob
import os
import sys
sys.path.append(os.path.abspath('.'))
from src.data_processing import doc_cleaning, chunking_doc
from src.vector_store import store_embeddings
DATA_DIR = "data"
LOG_FILE = "processed_log.txt"

def get_processed_files():
    if not os.path.exists(LOG_FILE):
        return set()
    with open(LOG_FILE, "r", encoding= "utf-8") as f:
        return set(line.strip() for line in f)

def log_processed_file(file_path):
    with open(LOG_FILE, "a", encoding= "utf-8") as f:
        f.write(f"{file_path}\n")

def main():
    if not os.path.exists(DATA_DIR):
        print(f"Lỗi: Thư mục {DATA_DIR} không tồn tại rồi.")
        return
    #quét và update data
    processed_files = get_processed_files()
    all_current_files = glob.glob(f"{DATA_DIR}/*.txt")

    new_files_process = [f for f in all_current_files if f not in processed_files]
    
    
    if not new_files_process:
        print("Không có file mới cần xử lí")
        return

    print(f"Tìm thấy {len(new_files_process)} file mới: {new_files_process}")

    
    all_chunks = []
    for file_path in glob.glob(f"{DATA_DIR}/*.txt"):
        print(f"Đang xử lí file {file_path}...")
        clean_text = doc_cleaning(file_path)
        chunks = chunking_doc(clean_text, file_path=file_path)
        all_chunks.extend(chunks)

    if all_chunks:
        print(f"Đã xử lí xong {len(all_chunks)} chunks. Bắt đầu embed vector store")
        store_embeddings(all_chunks)
        print(f"Lưu vào vector store thành công!")

    else:
        print("Không tìm thấy file nào để xử lí.")

    for file_path in new_files_process:
        log_processed_file(file_path)

    print("Quá trình cập nhật hoàn tất")

if __name__ == "__main__":
    main()


