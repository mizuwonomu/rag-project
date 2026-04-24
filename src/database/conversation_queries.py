def get_user_conversations(conn, user_id: str):
    with conn.cursor() as cur:
        #truy cập vào bảng conversations để lấy title 
        #title được tạo ra với mỗi hội thoại, tức tên cuộc hội thoại 
        cur.execute(
            "SELECT conversation_id, title FROM conversations WHERE user_id = %s ORDER BY created_at DESC",
            (user_id,)
        ) #lấy conversation gần nhất được tạo

        return cur.fetchall()