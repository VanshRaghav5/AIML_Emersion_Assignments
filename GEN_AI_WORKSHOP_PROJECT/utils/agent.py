from __future__ import annotations

from typing import Any, Dict

from langchain_groq import ChatGroq

from config import Settings
from .prompts import tutor_prompt_template


def get_llm(settings: Settings) -> ChatGroq:
    return ChatGroq(
        groq_api_key=settings.groq_api_key,
        model_name="llama-3.3-70b-versatile",
        temperature=0.2,
    )


def _format_context(chunks: list[str] | None, fallback: str = "") -> str:
    if not chunks:
        return fallback
    return "\n\n".join(chunks)


def route_and_run(
    *,
    feature: str,
    settings: Settings,
    subject: str,
    student_level: str,
    question: str | None = None,
    context_chunks: list[str] | None = None,
    available_days: int | None = None,
    hours_per_day: int | None = None,
    difficulty: str | None = None,
    num_questions: int = 5,
    quiz_results: str | None = None,
    practice_type: str | None = None,
) -> Dict[str, Any]:
    llm = get_llm(settings)
    context = _format_context(context_chunks)

    if feature == "Ask Questions":
        if not question:
            raise ValueError("Question is required")
        chain = tutor_prompt_template() | llm
        response = chain.invoke(
            {
                "student_level": student_level,
                "subject": subject,
                "question": question,
                "context": context,
            }
        )
        return {"output": response.content}

    if feature == "Study Plan":
        if available_days is None or hours_per_day is None:
            raise ValueError("available_days and hours_per_day are required")
        from .study_plan import generate_study_plan
        plan = generate_study_plan(
            llm,
            subject=subject,
            student_level=student_level,
            available_days=available_days,
            hours_per_day=hours_per_day,
            context=context,
        )
        return {"output": plan}

    if feature == "Quiz":
        if not difficulty:
            raise ValueError("Difficulty is required")
        from .quiz_generator import generate_quiz
        quiz = generate_quiz(
            settings,
            subject=subject,
            difficulty=difficulty,
            num_questions=num_questions,
            context=context,
        )
        return {"output": quiz}

    if feature == "Weak Area Analysis":
        if not quiz_results:
            raise ValueError("Quiz results are required")
        from .weak_area import analyze_weak_areas
        analysis = analyze_weak_areas(
            settings,
            subject=subject,
            student_level=student_level,
            quiz_results=quiz_results,
            context=context,
        )
        return {"output": analysis}

    if feature == "Practice Questions":
        if not difficulty or not practice_type:
            raise ValueError("Difficulty and practice_type are required")
        from .practice_questions import generate_practice_questions
        questions = generate_practice_questions(
            llm,
            subject=subject,
            difficulty=difficulty,
            practice_type=practice_type,
            num_questions=num_questions,
            context=context,
        )
        return {"output": questions}

    raise ValueError(f"Unknown feature: {feature}")
