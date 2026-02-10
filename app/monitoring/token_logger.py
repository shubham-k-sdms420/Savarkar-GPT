"""
Token Usage Logger.

Appends one JSON line per LLM request to logs/token_usage.jsonl.
Each entry records: timestamp, question, model, input/output/total tokens, latency.

Zero external dependencies — uses only stdlib json and pathlib.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from app.config.settings import settings

_LOG_DIR = settings.PROJECT_ROOT / "logs"
_LOG_FILE = _LOG_DIR / "token_usage.jsonl"


def log_token_usage(
    question: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    total_tokens: int,
    latency_ms: int,
) -> dict:
    """
    Append a single token-usage record to the JSONL log file.

    Returns the logged record dict.
    """
    _LOG_DIR.mkdir(parents=True, exist_ok=True)

    record = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
        "question": question,
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "latency_ms": latency_ms,
    }

    with open(_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    return record


def read_usage_logs() -> list[dict]:
    """Read all token usage records from the JSONL log file."""
    if not _LOG_FILE.exists():
        return []

    records = []
    with open(_LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def get_usage_summary() -> dict:
    """
    Return aggregate usage statistics.

    Returns dict with: total_requests, total_input_tokens, total_output_tokens,
    total_tokens, avg_input_tokens, avg_output_tokens, avg_latency_ms,
    and the last 10 requests.
    """
    records = read_usage_logs()

    if not records:
        return {
            "total_requests": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_tokens": 0,
            "avg_input_tokens": 0,
            "avg_output_tokens": 0,
            "avg_latency_ms": 0,
            "recent_requests": [],
        }

    total_input = sum(r.get("input_tokens", 0) for r in records)
    total_output = sum(r.get("output_tokens", 0) for r in records)
    total_tokens = sum(r.get("total_tokens", 0) for r in records)
    total_latency = sum(r.get("latency_ms", 0) for r in records)
    count = len(records)

    return {
        "total_requests": count,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_tokens": total_tokens,
        "avg_input_tokens": round(total_input / count),
        "avg_output_tokens": round(total_output / count),
        "avg_latency_ms": round(total_latency / count),
        "recent_requests": records[-10:],
    }
