from src.data_processing import doc_cleaning, chunking_doc

file_path = "data\\summary.txt"

def main():
    cleaned_text = doc_cleaning(file_path)

    print("\n---Sau khi cleaning (300 kí tự):")
    print(cleaned_text[:300])
    print("-----------------\n")

    text_chunks = chunking_doc(cleaned_text)

    print("\n--Hai chunk đầu tiên----")
    if (len(text_chunks) > 0):
        print("---Chunk 1---")
        print((text_chunks[0]))
    if (len(text_chunks) > 1):
        print("---Chunk 2---")
        print((text_chunks[1]))

    print("--------------------")

if __name__ == "__main__":
    main()

