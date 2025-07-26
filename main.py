from src.data_processing import doc_cleaning, chunking_doc
from src.vector_store import store_embeddings
file_path = "data/summary.txt"

def main():
    cleaned_text = doc_cleaning(file_path)
    
    text_chunks = chunking_doc(cleaned_text)

    vector_store = store_embeddings(text_chunks)

    print("\nToàn bộ quá trình đã xong")

    print("\nChecking vector store:...")

    query = "Kế hoạch A và B của NASA là gì"

    results = vector_store.similarity_search(query)
    print(f"\nTìm thấy {len(results)} chunks liên quan đến {query}")
    print("---Nội dung chunk đầu tiên tìm được---")

    if(results):
        print(results[0].page_content)
    else:
        print("Không tìm thấy chunk nào phù hợp")

if __name__ == "__main__":
    main()

