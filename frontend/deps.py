from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class AppDeps:
    rag_chain: Any
    db_connection_factory: Callable[[], Any]
    title_generator: Callable[[str, str], str]
    background_scheduler: Callable[..., None]
