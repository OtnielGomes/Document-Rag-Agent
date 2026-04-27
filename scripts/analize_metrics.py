import json
from statistics import mean
from pathlib import Path

from src.config import METRICS_FILE


def load_runs(path: Path) -> list[dict]:
    runs = []

    if not path.exists():
        raise FileNotFoundError(f"Log file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                runs.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[Warn] Invalid JSON on line {line_number}: {e}")

    return runs


def safe_mean(values: list[float]) -> float:
    numeric_values = [v for v in values if isinstance(v, (int, float))]
    return round(mean(numeric_values), 3) if numeric_values else 0.0


def analyze_runs(runs: list[dict]) -> dict:
    total_runs = len(runs)

    retrieval_latencies = [
        run.get("retrieval_latency", 0)
        for run in runs
        if isinstance(run.get("retrieval_latency", 0), (int, float))
    ]

    generation_latencies = [
        run.get("generation_latency", 0)
        for run in runs
        if isinstance(run.get("generation_latency", 0), (int, float))
    ]

    total_latencies = [
        run.get("total_latency", 0)
        for run in runs
        if isinstance(run.get("total_latency", 0), (int, float))
    ]

    retrieved_docs_counts = [
        run.get("retrieved_docs_count", 0)
        for run in runs
        if isinstance(run.get("retrieved_docs_count", 0), (int, float))
    ]

    error_count = sum(
        1
        for run in runs
        if run.get("status") == "error"
    )

    success_count = total_runs - error_count
    error_rate = round((error_count / total_runs) * 100, 2) if total_runs else 0.0

    summary = {
        "log_path": str(METRICS_FILE),
        "total_runs": total_runs,
        "avg_retrieval_latency": safe_mean(retrieval_latencies),
        "avg_generation_latency": safe_mean(generation_latencies),
        "avg_total_latency": safe_mean(total_latencies),
        "avg_retrieved_docs": safe_mean(retrieved_docs_counts),
        "success_count": success_count,
        "error_count": error_count,
        "error_rate_percent": error_rate,
    }

    return summary


def print_summary(summary: dict) -> None:
    print("\n=== RAG Metrics Summary ===")
    print(f"Log file: {summary['log_path']}")
    print(f"Total runs: {summary['total_runs']}")
    print(f"Average retrieval latency: {summary['avg_retrieval_latency']}s")
    print(f"Average generation latency: {summary['avg_generation_latency']}s")
    print(f"Average total latency: {summary['avg_total_latency']}s")
    print(f"Average retrieved docs: {summary['avg_retrieved_docs']}")
    print(f"Success count: {summary['success_count']}")
    print(f"Error count: {summary['error_count']}")
    print(f"Error rate: {summary['error_rate_percent']}%")


def main():
    runs = load_runs(METRICS_FILE)
    summary = analyze_runs(runs)
    print_summary(summary)


if __name__ == "__main__":
    main()