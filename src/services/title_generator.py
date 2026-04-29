from __future__ import annotations
from typing import Any
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
            You are an AI model that create concise chat title for a chatbot system conversation. 
            Your task is to read the first query of user and inference LLM's response, then write a summarization title.
             
            Strict rules:
            - Output must be Vietnamese only.
            - Output must be at most 8 words (strictly below 9 words).
            - Return title text only.
            - Do not include explanation.
            - Do not wrap output in quotes, brackets, markdown, or punctuation-only wrappers.

            Examples:
                - user: Khi nào sẽ được mở lớp học phần rút gọn vậy?
                - ai: Theo điều 10, khoản 5, sinh viên sẽ được mở học phần rút gọn khi thoả mãn đồng thời các điều kiện sau...
                YOUR output: Điều kiện học phần rút gọn

                - user: Giờ t đang nợ 9 tín, thì có bị sao không?
                - ai: Theo điều 19, khoản 1, sinh viên có số tín chỉ không đạt trong học kỳ lớn hơn 8 sẽ bị nâng một mức cảnh báo học tập ...
                YOUR output: Cảnh báo học tập

                - user: Nếu học phần được 3.38 điểm, vậy theo thang 10 là bao nhiêu?
                - ai: Theo điều 12, khoản 7, dải điểm tương đương và công thức quy đổi là...
                YOUR output: Quy đổi điểm học phần
            """),
        
        ("human", 
         """First user query: {question}
            First AI answer: {full_response}

            Generate a single Vietnamese title that follows all rules.
         """), #truyền luôn response string của llm vào human để tránh missing fields
    ]
)

generate_title_llm = ChatGroq(
    model="llama-3.1-8b-instant",
        max_retries=0,
        temperature= 0.3
)
title_chain = prompt | generate_title_llm | StrOutputParser()

def _normalize_assistant_answer(full_response: Any) -> str:
    """Chuẩn hoá AIMessage từ inference response sang dạng string
    để làm đầu vào cho title generation"""
    if isinstance(full_response, str):
        return full_response
    
    if isinstance(full_response, AIMessage):
        return full_response.content if isinstance(full_response.content, str) else str(full_response.content)
    
    if isinstance(full_response, dict):
        for key in ("answer", "content", "output", "text"):
            value = full_response.get(key)
            if value:
                return str(value)
        return str(full_response)
    
    if isinstance(full_response, list):
        return " ".join(str(item) for item in full_response if item is not None)
    
    return str(full_response)


def generate_title(question: str, full_response: Any) -> str:
    """Tạo title ngắn gọn"""

    assistant_text = _normalize_assistant_answer(full_response).strip()
    result = title_chain.invoke(
        {
            "question": question.strip(),
            "full_response": assistant_text,
        }
    )

    return result.strip()

