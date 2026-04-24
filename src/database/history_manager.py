from langchain_postgres import PostgresChatMessageHistory

def get_postgres_history(conn, conversation_id: str) -> PostgresChatMessageHistory:
    #sử dụng thẳng conversation_id uuid4 để map với bảng conversations
    #với user_id
    #create history object
    history = PostgresChatMessageHistory(
        "chat_history",
        conversation_id,
        sync_connection=conn,
    )

    return history