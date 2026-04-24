import os
import psycopg
import streamlit as st
from psycopg.errors import OperationalError
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