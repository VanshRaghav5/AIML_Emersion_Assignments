from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from .prompts import study_plan_prompt_template


def generate_study_plan(
    llm: BaseChatModel,
    *,
    subject: str,
    student_level: str,
    available_days: int,
    hours_per_day: int,
    context: str,
) -> str:
    chain = study_plan_prompt_template() | llm
    response = chain.invoke(
        {
            "subject": subject,
            "student_level": student_level,
            "available_days": available_days,
            "hours_per_day": hours_per_day,
            "context": context,
        }
    )
    return str(response.content)
