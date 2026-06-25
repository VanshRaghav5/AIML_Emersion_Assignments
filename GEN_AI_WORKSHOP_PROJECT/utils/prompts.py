from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate


def tutor_prompt_template():
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert tutor. Use the provided context to answer the student's question.\n"
                "Explain the concepts step-by-step, adjust your language for the student's level, and provide examples.\n"
                "If the answer isn't in the context, mention that clearly and suggest what areas of the notes to review.",
            ),
            (
                "human",
                "Student Level: {student_level}\n"
                "Subject: {subject}\n\n"
                "Question: {question}\n\n"
                "Context Notes:\n{context}",
            ),
        ]
    )


def study_plan_prompt_template():
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a study coach. Create a clear, structured study plan matching the student's level, available days, and daily hours.\n"
                "Incorporate study topics, dedicated review slots, and practice recommendations. "
                "Base it on the notes context when available.",
            ),
            (
                "human",
                "Subject: {subject}\n"
                "Student Level: {student_level}\n"
                "Days Available: {available_days}\n"
                "Hours Per Day: {hours_per_day}\n\n"
                "Context Notes:\n{context}",
            ),
        ]
    )


def quiz_prompt_template():
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an exam prep assistant. Generate a multiple-choice quiz based on the provided notes.\n"
                "Return a raw JSON array of objects. Do not include markdown formatting or backticks. Each object must have these keys:\n"
                "- 'question': the question text\n"
                "- 'options': exactly 4 multiple-choice options\n"
                "- 'answer': the correct answer option (must match one of the options exactly)\n"
                "- 'explanation': a brief explanation of the correct answer.\n\n"
                "If notes are insufficient, generate best-effort questions and note in the explanation that they are inferred.",
            ),
            (
                "human",
                "Difficulty: {difficulty}\n"
                "Number of Questions: {num_questions}\n\n"
                "Subject: {subject}\n"
                "Context Notes:\n{context}",
            ),
        ]
    )


def weak_area_prompt_template():
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an educational analyst. Review the student's quiz responses and identify which concepts they understand well and which ones need improvement.\n"
                "You MUST return a JSON object with the following keys:\n"
                "- 'summary': a brief paragraph summarizing the overall performance and core recommendations.\n"
                "- 'concepts': a JSON array of objects, where each object has these keys:\n"
                "  - 'name': name of the concept\n"
                "  - 'score': integer from 0 to 100 representing mastery rating\n"
                "  - 'status': string (must be 'Strong', 'Moderate', or 'Weak')\n"
                "  - 'feedback': actionable next steps to review this concept.\n\n"
                "Only return a valid raw JSON object. Do not wrap the JSON output in markdown blocks or backticks.",
            ),
            (
                "human",
                "Subject: {subject}\n"
                "Student Level: {student_level}\n\n"
                "Quiz Results:\n{quiz_results}\n\n"
                "Context Notes:\n{context}",
            ),
        ]
    )


def practice_prompt_template():
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a question designer. Generate progressive practice questions based on the notes context.\n"
                "Include detailed explanations. If the notes are insufficient, clarify what was inferred.",
            ),
            (
                "human",
                "Subject: {subject}\n"
                "Difficulty: {difficulty}\n"
                "Practice Style: {practice_type}\n"
                "Number of Questions: {num_questions}\n\n"
                "Context Notes:\n{context}",
            ),
        ]
    )
