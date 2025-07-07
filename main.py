from src.data_processing import chunk_text, load_and_clean_content

file_path = "D:\\rag-project\\data\\summary.txt"

chunk_size = 500

def main():
    cleaned_text = load_and_clean_content(file_path)

    print("\n----Sau khi clean-----------")
    print(cleaned_text[:300])
    print("------------------------\n")

    chunked = chunk_text(cleaned_text,chunk_size)
    print("\n--Hai chunk đầu tiên--")

    if len(chunked) > 0:
        print("--- Chunk 1 ---")
        print(chunked[0])
    if len(chunked) > 1:
        print("--- Chunk 2 ---")
        print(chunked[1])
    
    print("------------------")

if __name__ == "__main__":
    main()


