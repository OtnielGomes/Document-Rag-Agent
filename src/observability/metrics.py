from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any, Dict

from src.config import METRICS_FILE


def ensure_log_dir() -> None:
    METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)


def build_run_log(
    *,
    question: str,
    source: str,
    answer: str,
    retrieved_docs_count: int,
    retrieval_latency: float,
    generation_latency: float,
    total_latency: float,
    status: str = "success",
    error: str | None = None,
) -> Dict[str, Any]:
    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "question": question,
        "source": source,
        "answer": answer,
        "retrieved_docs_count": retrieved_docs_count,
        "retrieval_latency": retrieval_latency,
        "generation_latency": generation_latency,
        "total_latency": total_latency,
        "status": status,
        "error": error,
    }


def save_run_log(log_data: Dict[str, Any]) -> None:
    ensure_log_dir()

    with METRICS_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(log_data, ensure_ascii=False) + "\n")