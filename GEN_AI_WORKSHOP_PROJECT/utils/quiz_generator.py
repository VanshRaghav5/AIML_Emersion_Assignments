from __future__ import annotations

import json
from typing import Any, Dict, List

from langchain_groq import ChatGroq

from config import Settings
from .prompts import quiz_prompt_template


def get_quiz_llm(settings: Settings) -> ChatGroq:
    return ChatGroq(
        groq_api_key=settings.groq_api_key,
        model_name="llama-3.3-70b-versatile",
        temperature=0.2,
        model_kwargs={"response_format": {"type": "json_object"}},
    )


def generate_quiz(
    settings: Settings,
    *,
    subject: str,
    difficulty: str,
    num_questions: int,
    context: str,
) -> List[Dict[str, Any]]:
    llm = get_quiz_llm(settings)
    chain = quiz_prompt_template() | llm
    response = chain.invoke(
        {
            "difficulty": difficulty,
            "num_questions": num_questions,
            "subject": subject,
            "context": context,
        }
    )

    try:
        data = json.loads(str(response.content))
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "questions" in data:
            return data["questions"]
        return [data]
    except Exception as e:
        return [
            {
                "question": "Failed to parse quiz response.",
                "options": [
                    "JSON parsing error",
                    "Try regenerating",
                    "Check API Key configuration",
                    "Check note size",
                ],
                "answer": "JSON parsing error",
                "explanation": f"Failed with error: {e}. Raw response: {response.content}",
            }
        ]
