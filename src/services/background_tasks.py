import logging
import threading
from concurrent.futures import ThreadPoolExecutor

import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

logger = logging.getLogger(__name__)

#only create threadpool one time only
@st.cache_resource
def get_background_pool() -> ThreadPoolExecutor:
    return ThreadPoolExecutor(max_workers=2) #tạo ra 2 luồng chính chạy song song
                                             #tức 1 bên chạy generate title, người dùng vẫn có thể tiếp tục chat
                                             #2 thread sẽ không cần phải chờ nhau

#streamlit luôn chạy theo hướng single-threaded, luôn chạy lại từ đầu đến cuối file

def fire_and_forget(func, *args, **kwargs):
    """Gọi callable và trả về task tương ứng"""
    ctx = get_script_run_ctx() #st.session_state luôn được quản lý dựa trên biến context của từng tab
    #khi tạo một thread chạy ngầm, nếu không có context -> sẽ không thể biết st.session_state hiện tại
    #dùng get_script để copy context của user hiện tại vào main thread

    def _run_with_context():
        try:
            thread = threading.current_thread()
            try:
                add_script_run_ctx(thread, ctx) #từ đó gán thẻ context vào thread chạy ngầm
            except TypeError:
                #fallback nếu các ver của streamlit chỉ chấp nhận 1 thread
                add_script_run_ctx(thread)

            return func(*args, **kwargs)
        except Exception:
            logger.exception("Unhandled exception in background tasks")
            return None
        
    try:
        pool = get_background_pool()
        return pool.submit(_run_with_context) 
    except Exception:
        logger.exception("Failed to submit background task")
        return None