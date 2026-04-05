import asyncio
import os
from pathlib import Path

from llama_cloud import AsyncLlamaCloud
from dotenv import load_dotenv

load_dotenv()
PDF_PATH = "data_quyche/QCDT_2025_DHBK.pdf"

async def parser() -> None:
    client = AsyncLlamaCloud(api_key=os.environ.get("LLAMA_API_KEY"))

    file_obj = await client.files.create(file=PDF_PATH, purpose="parse")

    result = await client.parsing.parse(
        file_id= file_obj.id,
        tier="agentic",
        version="latest",
        agentic_options={
        "custom_prompt": "- Do not output main parent header \"Chương\" as type \"##\", must replace each of it with \"#\" header level\n- Do not output each header named \\\"Điều\\\" as BOLD MARKDOWN TYPE (****), instead replace it with ### header level. This act as a child header inherit from parent header \"Chương\""
        },
        output_options={
            "markdown": {
                "tables": {
                    "output_tables_as_markdown": True
                }
            }
        },
        expand="markdown_full",
    )

    markdown_full = result.markdown_full or ""
    out_path = Path(PDF_PATH).with_suffix(".md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(markdown_full, encoding="utf-8")

if __name__ == "__main__":
    asyncio.run(parser())
