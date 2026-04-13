import os
import psycopg
import streamlit as st
import uuid
from psycopg.errors import OperationalError
from langchain_postgres import PostgresChatMessageHistory
from dotenv import load_dotenv

load_dotenv()

#check if connection still working - if not, connect cái mới vứt đi cache connect cũ
def is_connection_alive(conn) -> bool:
    try:
        if conn.closed:
            return False
        #nếu connection có vẻ mở nhưng db server đã sập
        conn.execute("SELECT 1")
        return True

    except (OperationalError, psycopg.Error):
        return False



#cache file streamlit để tránh connection mỗi lần mở hay chat input
@st.cache_resource(validate=is_connection_alive)
def get_db_connection():
    DATABASE_URL = os.environ.get("DATABASE_URL")

    conn = psycopg.connect(DATABASE_URL)
    return conn


def get_postgres_history(conversation_id: str) -> PostgresChatMessageHistory:
    #sử dụng thẳng conversation_id uuid4 để map với bảng conversations
    #với user_id
    conn = get_db_connection()

    #create history object
    history = PostgresChatMessageHistory(
        "chat_history",
        conversation_id,
        sync_connection=conn,
    )

    return history


def get_user_conversations(user_id: str):
    conn = get_db_connection
    with conn.cursor() as cur:
        #truy cập vào bảng conversations để lấy title 
        #title được tạo ra với mỗi hội thoại, tức tên cuộc hội thoại 
        cur.execute(
            "SELECT conversation_id, title FROM conversations WHERE user_id = %s ORDER BY created_at DESC",
            (user_id,)
        ) #lấy conversation gần nhất được tạo

        return cur.fetchall()