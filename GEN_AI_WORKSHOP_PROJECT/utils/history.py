from __future__ import annotations

from datetime import datetime
import json
import os
from typing import Any, Dict, List


def _get_history_path() -> str:
    os.makedirs("vector_store", exist_ok=True)
    return "vector_store/history.json"


def load_history() -> List[Dict[str, Any]]:
    path = _get_history_path()
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception:
        return []


def save_quiz_record(
    *,
    subject: str,
    score: int,
    concepts: List[Dict[str, Any]],
    summary: str,
) -> None:
    history = load_history()
    record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "subject": subject,
        "score": score,
        "concepts": concepts,
        "summary": summary,
    }
    history.append(record)
    path = _get_history_path()
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Failed to write history: {e}")


def get_history_metrics() -> Dict[str, Any]:
    history = load_history()
    if not history:
        return {
            "total_quizzes": 0,
            "average_score": 0,
            "weakest_concept": "None",
        }

    total_quizzes = len(history)
    total_score = sum(r["score"] for r in history)
    average_score = int(total_score / total_quizzes)

    concept_totals: Dict[str, List[int]] = {}
    for r in history:
        for c in r.get("concepts", []):
            name = c.get("name", "Unknown")
            score = c.get("score", 100)
            concept_totals.setdefault(name, []).append(score)

    weakest_concept = "None"
    lowest_avg = 101.0
    for name, scores in concept_totals.items():
        avg = sum(scores) / len(scores)
        if avg < lowest_avg:
            lowest_avg = avg
            weakest_concept = name

    return {
        "total_quizzes": total_quizzes,
        "average_score": average_score,
        "weakest_concept": weakest_concept,
    }
