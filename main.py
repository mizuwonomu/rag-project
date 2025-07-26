from src.qa_chain import get_chain

def main():
    rag_chain = get_chain()
    print("RAG chain is ready. Đặt câu hỏi của m đi")

    while True:
        question = input(">> ")

        if question.lower() == 'exit':
            break


        answer = rag_chain.invoke(question)
        print("\nTrả lời:", answer, "\n")

if __name__ == "__main__":
    main()

    