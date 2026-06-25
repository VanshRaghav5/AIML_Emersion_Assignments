from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from .prompts import practice_prompt_template


def generate_practice_questions(
    llm: BaseChatModel,
    *,
    subject: str,
    difficulty: str,
    practice_type: str,
    num_questions: int,
    context: str,
) -> str:
    chain = practice_prompt_template() | llm
    response = chain.invoke(
        {
            "subject": subject,
            "difficulty": difficulty,
            "practice_type": practice_type,
            "num_questions": num_questions,
            "context": context,
        }
    )
    return str(response.content)
