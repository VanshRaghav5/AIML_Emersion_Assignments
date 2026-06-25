from __future__ import annotations

import json
from typing import Any, Dict

from langchain_groq import ChatGroq

from config import Settings
from .prompts import weak_area_prompt_template


def get_weak_area_llm(settings: Settings) -> ChatGroq:
    """Instantiate a specialized Chat model configured for structured JSON output."""
    return ChatGroq(
        groq_api_key=settings.groq_api_key,
        model_name="llama-3.3-70b-versatile",
        temperature=0.2,
        model_kwargs={"response_format": {"type": "json_object"}},
    )


def analyze_weak_areas(
    settings: Settings,
    *,
    subject: str,
    student_level: str,
    quiz_results: str,
    context: str,
) -> Dict[str, Any]:
    """Analyze student quiz results to highlight weak concepts and suggest revisions."""
    llm = get_weak_area_llm(settings)
    chain = weak_area_prompt_template() | llm
    response = chain.invoke(
        {
            "subject": subject,
            "student_level": student_level,
            "quiz_results": quiz_results,
            "context": context,
        }
    )

    try:
        return json.loads(str(response.content))
    except Exception as e:
        return {
            "summary": f"Analysis JSON parsing error: {e}",
            "concepts": [
                {
                    "name": "Performance Review",
                    "score": 50,
                    "status": "Moderate",
                    "feedback": f"Please review notes manually. Raw output: {response.content}",
                }
            ],
        }
