from pathlib import Path

from langchain_text_splitters import MarkdownHeaderTextSplitter

_REPO_ROOT = Path(__file__).resolve().parents[2]
MARKDOWN_PATH = _REPO_ROOT / "data_quyche" / "QCDT_2025_DHBK.md"

markdown_documents: str = MARKDOWN_PATH.read_text(encoding="utf-8")

headers_to_split_on = [
    ("#", "Chương"),
    ("###", "Điều"),
]

markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

markdown_chunk_documents = markdown_splitter.split_text(markdown_documents)