from __future__ import annotations

import hashlib
import os
from typing import Optional

import pandas as pd
import streamlit as st

from config import get_settings, Settings
from utils.agent import route_and_run
from utils.pdf_loader import extract_text_from_pdf
from utils.retriever import (
    build_faiss_index,
    get_retriever,
    load_faiss_index,
    save_faiss_index,
    split_text,
)
from utils.history import load_history, save_quiz_record, get_history_metrics

st.set_page_config(page_title="GenAI Learning Mentor", layout="wide")


def _get_context_chunks(settings, *, faiss_index, question: str) -> list[str]:
    retriever = get_retriever(faiss_index, k=settings.k_retrieval)
    docs = retriever.invoke(question)
    if not docs:
        return []
    return [d.page_content for d in docs]


def _ensure_vector_store_for_pdf(settings, file_bytes: bytes, *, force_rebuild: bool):
    os.makedirs("vector_store", exist_ok=True)
    meta_path = "vector_store/last_pdf.sha256"

    file_hash = hashlib.sha256(file_bytes).hexdigest()
    last_hash: Optional[str] = None
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            last_hash = f.read().strip() or None

    needs_rebuild = force_rebuild or (last_hash != file_hash) or (not os.path.exists(settings.faiss_index_path))

    with st.spinner("Building embeddings & FAISS index from PDF..." if needs_rebuild else "Loading existing FAISS index..."):
        if not needs_rebuild:
            index = load_faiss_index(settings)
            return index, file_hash, False

        text = extract_text_from_pdf(file_bytes)
        if not text or not text.strip():
            raise ValueError("Empty PDF: no extractable text found.")

        docs = split_text(text, chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
        if not docs:
            raise ValueError("No text chunks generated from PDF.")

        index = build_faiss_index(settings=settings, documents=docs)
        save_faiss_index(index, settings)
        with open(meta_path, "w", encoding="utf-8") as f:
            f.write(file_hash)

    return index, file_hash, True


def main():
    # Inject Custom CSS for premium glassmorphism dark-theme
    st.markdown(
        """
        <style>
        /* Import Google Font */
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

        /* Global styling overrides */
        html, body, [data-testid="stAppViewContainer"] {
            font-family: 'Outfit', sans-serif;
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            color: #f8fafc;
        }
        
        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: rgba(15, 23, 42, 0.95) !important;
            border-right: 1px solid rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
        }
        
        /* Title Styling */
        h1 {
            font-family: 'Outfit', sans-serif;
            font-weight: 700 !important;
            background: linear-gradient(to right, #6366f1, #a855f7);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 5px !important;
        }

        /* Styled Cards */
        .card {
            background: rgba(30, 41, 59, 0.5);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 18px;
            margin-bottom: 16px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            backdrop-filter: blur(8px);
        }

        .quiz-card {
            border-left: 4px solid #6366f1;
        }

        .correct-card {
            border-left: 4px solid #10b981;
            background: rgba(16, 185, 129, 0.08);
        }

        .incorrect-card {
            border-left: 4px solid #ef4444;
            background: rgba(239, 68, 68, 0.08);
        }

        /* Styled Tab headers */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background-color: rgba(15, 23, 42, 0.4);
            padding: 6px;
            border-radius: 10px;
            border: 1px solid rgba(255, 255, 255, 0.05);
        }

        .stTabs [data-baseweb="tab"] {
            height: 40px;
            white-space: pre-wrap;
            background-color: transparent;
            border-radius: 6px;
            color: #94a3b8;
            font-weight: 500;
            padding: 0px 16px;
            transition: all 0.3s ease;
        }

        .stTabs [data-baseweb="tab"]:hover {
            color: #f8fafc;
            background-color: rgba(255, 255, 255, 0.03);
        }

        .stTabs [aria-selected="true"] {
            background-color: #6366f1 !important;
            color: #ffffff !important;
            box-shadow: 0 4px 10px rgba(99, 102, 241, 0.3);
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("GenAI Learning Mentor")
    st.caption("RAG-based personalized learning assistant powered by Google Gemini and Groq")

    settings = get_settings()
    google_key = settings.google_api_key
    groq_key = settings.groq_api_key

    with st.sidebar:
        st.header("API Configuration")
        
        # Google API Key setup (for embeddings)
        if not google_key:
            google_key = st.text_input(
                "Enter Google API Key",
                type="password",
                placeholder="AIzaSy...",
                help="Enter your Gemini API Key for PDF indexing.",
            )
            if not google_key:
                st.warning("Google API Key missing. PDF upload disabled.")
        else:
            st.success("Google API Key detected.")

        # Groq API Key setup (for LLM generation)
        if not groq_key:
            groq_key = st.text_input(
                "Enter Groq API Key",
                type="password",
                placeholder="gsk_...",
                help="Enter your Groq API Key for LLM generations.",
            )
            if not groq_key:
                st.warning("Groq API Key missing. LLM generation disabled.")
        else:
            st.success("Groq API Key detected.")

        # Re-build settings object if keys are provided via UI
        if google_key or groq_key:
            settings = Settings(
                google_api_key=google_key or None,
                groq_api_key=groq_key or None,
                faiss_index_path=settings.faiss_index_path,
                faiss_docs_path=settings.faiss_docs_path,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
                k_retrieval=settings.k_retrieval,
            )

        st.divider()
        st.header("Course Notes")
        uploaded_files = st.file_uploader(
            "Upload one or more PDFs", type=["pdf"], accept_multiple_files=True
        )
        force_rebuild = st.checkbox("Rebuild vector store from uploaded PDFs", value=False)

        st.divider()
        st.header("Student Profile")
        student_level = st.selectbox("Student level", ["Beginner", "Intermediate", "Advanced"], index=1)
        subject = st.text_input("Subject", value="General")

        st.divider()
        available_days = st.number_input("Days available", min_value=1, max_value=365, value=7)
        hours_per_day = st.number_input("Study hours per day", min_value=1, max_value=12, value=2)

    file_bytes_list = []
    if uploaded_files:
        for pdf_file in uploaded_files:
            file_bytes_list.append(pdf_file.getvalue())

    vector_index = None

    if file_bytes_list:
        if not google_key:
            st.sidebar.error("Google API Key required to process uploaded notes.")
        else:
            combined = b"".join(file_bytes_list)
            try:
                vector_index, file_hash, rebuilt = _ensure_vector_store_for_pdf(
                    settings,
                    combined,
                    force_rebuild=force_rebuild,
                )
                st.sidebar.success(f"Vector store ready (PDF hash: {file_hash[:10]}...)")
            except Exception as e:
                st.sidebar.error(str(e))
                vector_index = None
    else:
        if google_key and os.path.exists(settings.faiss_index_path):
            try:
                vector_index = load_faiss_index(settings)
                if vector_index:
                    st.sidebar.info("Loaded cached notes vector store from disk.")
            except Exception as e:
                st.sidebar.warning(f"Could not load cached index: {e}")

    tabs = st.tabs(
        ["Ask Questions", "Study Plan", "Quiz", "Weak Area Analysis", "Practice Questions", "Learning History"]
    )

    with tabs[0]:
        st.subheader("RAG Question Answering")
        question = st.text_area("Enter your question", height=120, placeholder="e.g., Explain the key ideas of supervised learning.")
        if st.button("Get Answer", type="primary", disabled=not question):
            if not groq_key:
                st.error("Groq API Key is required. Please configure it in the sidebar.")
            elif not vector_index:
                st.error("Missing vector store. Upload a PDF first.")
            else:
                with st.spinner("Retrieving relevant notes & generating answer..."):
                    context_chunks = _get_context_chunks(settings, faiss_index=vector_index, question=question)
                    response = route_and_run(
                        feature="Ask Questions",
                        settings=settings,
                        subject=subject,
                        student_level=student_level,
                        question=question,
                        context_chunks=context_chunks,
                    )
                st.markdown("### Answer")
                st.markdown(response["output"])

    with tabs[1]:
        st.subheader("Personalized Study Plan")
        if st.button("Generate Study Plan", type="primary"):
            if not groq_key:
                st.error("Groq API Key is required. Please configure it in the sidebar.")
            elif not vector_index:
                st.error("Missing vector store. Upload a PDF first.")
            else:
                with st.spinner("Generating study roadmap..."):
                    context_chunks = _get_context_chunks(settings, faiss_index=vector_index, question=f"Create study topics for {subject}")
                    response = route_and_run(
                        feature="Study Plan",
                        settings=settings,
                        subject=subject,
                        student_level=student_level,
                        available_days=int(available_days),
                        hours_per_day=int(hours_per_day),
                        context_chunks=context_chunks,
                    )
                st.markdown(response["output"])

    with tabs[2]:
        st.subheader("Quiz Generator")
        
        if "quiz_questions" not in st.session_state:
            difficulty = st.selectbox("Difficulty", ["Easy", "Medium", "Hard"], index=1, key="quiz_diff")
            num_questions = st.number_input("Number of questions", min_value=3, max_value=30, value=5, key="quiz_count")
            
            if st.button("Generate Quiz", type="primary", key="gen_quiz_btn"):
                if not groq_key:
                    st.error("Groq API Key is required. Please configure it in the sidebar.")
                elif not vector_index:
                    st.error("Missing vector store. Upload a PDF first.")
                else:
                    with st.spinner("Generating interactive quiz from your notes..."):
                        context_chunks = _get_context_chunks(settings, faiss_index=vector_index, question=f"Generate a {difficulty} MCQ quiz on {subject}")
                        response = route_and_run(
                            feature="Quiz",
                            settings=settings,
                            subject=subject,
                            student_level=student_level,
                            difficulty=difficulty,
                            num_questions=int(num_questions),
                            context_chunks=context_chunks,
                        )
                        if response and "output" in response:
                            st.session_state["quiz_questions"] = response["output"]
                            st.session_state["quiz_submitted"] = False
                            st.session_state["quiz_answers"] = {}
                            if "active_weak_analysis" in st.session_state:
                                del st.session_state["active_weak_analysis"]
                            st.rerun()
        else:
            questions = st.session_state["quiz_questions"]
            submitted = st.session_state["quiz_submitted"]
            
            st.info(f"Quiz on **{subject}** ({len(questions)} Questions) - {student_level} Level")
            
            answers = {}
            for i, q in enumerate(questions):
                card_class = "card quiz-card"
                if submitted:
                    user_ans = st.session_state["quiz_answers"].get(f"q_{i}")
                    if user_ans == q["answer"]:
                        card_class = "card correct-card"
                    else:
                        card_class = "card incorrect-card"
                
                st.markdown(f'<div class="{card_class}"><b>Question {i+1}:</b> {q["question"]}</div>', unsafe_allow_html=True)
                
                if not submitted:
                    answers[f"q_{i}"] = st.radio(
                        "Select your answer:",
                        options=q["options"],
                        key=f"radio_q_{i}",
                        index=None,
                    )
                else:
                    user_ans = st.session_state["quiz_answers"].get(f"q_{i}")
                    correct_ans = q["answer"]
                    
                    st.markdown(f"**Your Answer:** `{user_ans or 'No Answer Selected'}`")
                    st.markdown(f"**Correct Answer:** `{correct_ans}`")
                    st.markdown(f"*Explanation:* {q['explanation']}")
                    st.divider()

            if not submitted:
                if st.button("Submit Quiz", type="primary", key="submit_quiz_btn"):
                    st.session_state["quiz_answers"] = answers
                    
                    # Grade the quiz
                    correct_count = 0
                    results_str = ""
                    for i, q in enumerate(questions):
                        user_ans = answers.get(f"q_{i}")
                        status = "Correct" if user_ans == q["answer"] else "Incorrect"
                        if user_ans == q["answer"]:
                            correct_count += 1
                        results_str += f"Question {i+1}: {q['question']}\n- User answer: {user_ans}\n- Correct answer: {q['answer']}\n- Status: {status}\n\n"

                    score_pct = int((correct_count / len(questions)) * 100)

                    # Trigger Automatic Weak Area Analysis in the background
                    with st.spinner("Analyzing quiz responses and mapping weak areas..."):
                        context_chunks = _get_context_chunks(settings, faiss_index=vector_index, question=f"Weak concepts for {subject}")
                        analysis_res = route_and_run(
                            feature="Weak Area Analysis",
                            settings=settings,
                            subject=subject,
                            student_level=student_level,
                            quiz_results=results_str,
                            context_chunks=context_chunks,
                        )
                        analysis_data = analysis_res["output"]
                        st.session_state["active_weak_analysis"] = analysis_data
                    
                    # Store record locally to history database
                    save_quiz_record(
                        subject=subject,
                        score=score_pct,
                        concepts=analysis_data.get("concepts", []),
                        summary=analysis_data.get("summary", ""),
                    )

                    st.session_state["quiz_submitted"] = True
                    st.rerun()
            else:
                correct_count = 0
                for i, q in enumerate(questions):
                    if st.session_state["quiz_answers"].get(f"q_{i}") == q["answer"]:
                        correct_count += 1
                
                score_pct = int((correct_count / len(questions)) * 100)
                st.subheader(f"Quiz Results: {correct_count}/{len(questions)} ({score_pct}%)")
                st.progress(correct_count / len(questions))
                
                if score_pct >= 80:
                    st.success("Excellent performance! You have mastered these notes.")
                elif score_pct >= 50:
                    st.warning("Good job! A little more review will help you master the material.")
                else:
                    st.error("Consider reviewing the notes again and retaking the quiz.")

                # Render Automatic Weak Area Analysis Visualizations
                if "active_weak_analysis" in st.session_state:
                    st.divider()
                    st.subheader("💡 Automated Weak Area Analysis")
                    analysis_data = st.session_state["active_weak_analysis"]
                    
                    st.write(analysis_data.get("summary", ""))
                    
                    concepts = analysis_data.get("concepts", [])
                    if concepts:
                        df_concepts = pd.DataFrame([
                            {"Concept": c["name"], "Mastery Score": c["score"]} for c in concepts
                        ])
                        st.markdown("##### Concept Mastery Score Chart")
                        st.bar_chart(df_concepts, x="Concept", y="Mastery Score", color="#6366f1")

                        st.markdown("##### Topic Recommendations & Action Steps")
                        for c in concepts:
                            status = c.get("status", "Moderate")
                            card_color = "#10b981" if status == "Strong" else "#f59e0b" if status == "Moderate" else "#ef4444"
                            bg_color = "rgba(16, 185, 129, 0.05)" if status == "Strong" else "rgba(245, 158, 11, 0.05)" if status == "Moderate" else "rgba(239, 68, 68, 0.05)"
                            
                            st.markdown(
                                f"""
                                <div style="border-left: 4px solid {card_color}; background-color: {bg_color}; padding: 14px; border-radius: 8px; margin-bottom: 12px; border: 1px solid rgba(255, 255, 255, 0.05);">
                                    <b style="color: {card_color};">{status.upper()}</b> | <b>{c['name']} ({c['score']}%)</b><br>
                                    <span style="font-size: 14px; color: #cbd5e1;">{c['feedback']}</span>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                st.divider()
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Retake Quiz", key="retake_quiz_btn"):
                        st.session_state["quiz_submitted"] = False
                        st.session_state["quiz_answers"] = {}
                        if "active_weak_analysis" in st.session_state:
                            del st.session_state["active_weak_analysis"]
                        st.rerun()
                with col2:
                    if st.button("Generate New Quiz", key="new_quiz_btn"):
                        del st.session_state["quiz_questions"]
                        if "quiz_submitted" in st.session_state:
                            del st.session_state["quiz_submitted"]
                        if "quiz_answers" in st.session_state:
                            del st.session_state["quiz_answers"]
                        if "active_weak_analysis" in st.session_state:
                            del st.session_state["active_weak_analysis"]
                        st.rerun()

    with tabs[3]:
        st.subheader("Weak Area Identification")
        st.caption("Paste quiz results below (or raw answers) to get an automated concept analysis.")
        st.info("💡 Tip: Quizzes taken in the Quiz tab are analyzed automatically. Use this tab for manual review of outside tests.")
        quiz_results = st.text_area("Quiz results", height=160, placeholder="e.g., Q1 correct, Q2 incorrect (your answer: ...), ...")
        if st.button("Analyze Weak Areas", type="primary", disabled=not quiz_results.strip()):
            if not groq_key:
                st.error("Groq API Key is required. Please configure it in the sidebar.")
            elif not vector_index:
                st.error("Missing vector store. Upload a PDF first.")
            else:
                with st.spinner("Analyzing performance & suggesting revisions..."):
                    context_chunks = _get_context_chunks(settings, faiss_index=vector_index, question=f"Weak concepts for {subject} at level {student_level}")
                    response = route_and_run(
                        feature="Weak Area Analysis",
                        settings=settings,
                        subject=subject,
                        student_level=student_level,
                        quiz_results=quiz_results,
                        context_chunks=context_chunks,
                    )
                    analysis_data = response["output"]
                
                st.markdown("### Analysis Summary")
                st.write(analysis_data.get("summary", ""))
                
                concepts = analysis_data.get("concepts", [])
                if concepts:
                    df = pd.DataFrame([
                        {"Concept": c["name"], "Mastery Score": c["score"]} for c in concepts
                    ])
                    st.markdown("### Concept Mastery Levels")
                    st.bar_chart(df, x="Concept", y="Mastery Score", color="#6366f1")
                    
                    st.markdown("### Topic Recommendations")
                    for c in concepts:
                        status = c.get("status", "Moderate")
                        card_color = "#10b981" if status == "Strong" else "#f59e0b" if status == "Moderate" else "#ef4444"
                        bg_color = "rgba(16, 185, 129, 0.05)" if status == "Strong" else "rgba(245, 158, 11, 0.05)" if status == "Moderate" else "rgba(239, 68, 68, 0.05)"
                        st.markdown(
                            f"""
                            <div style="border-left: 4px solid {card_color}; background-color: {bg_color}; padding: 14px; border-radius: 8px; margin-bottom: 12px; border: 1px solid rgba(255, 255, 255, 0.05);">
                                <b>{status.upper()}</b> | <b>{c['name']} ({c['score']}%)</b><br>
                                <span style="font-size: 14px; color: #cbd5e1;">{c['feedback']}</span>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

    with tabs[4]:
        st.subheader("Practice Question Generator")
        difficulty = st.selectbox("Difficulty", ["Easy", "Medium", "Hard"], index=0)
        practice_type = st.selectbox(
            "Practice type",
            ["Application-based", "Interview questions", "Coding questions", "Conceptual questions"],
            index=0,
        )
        num_questions = st.number_input("Number of questions", min_value=3, max_value=30, value=8)

        if st.button("Generate Practice Set", type="primary"):
            if not groq_key:
                st.error("Groq API Key is required. Please configure it in the sidebar.")
            elif not vector_index:
                st.error("Missing vector store. Upload a PDF first.")
            else:
                with st.spinner("Generating practice questions..."):
                    context_chunks = _get_context_chunks(settings, faiss_index=vector_index, question=f"Generate {difficulty} {practice_type} practice questions for {subject}")
                    response = route_and_run(
                        feature="Practice Questions",
                        settings=settings,
                        subject=subject,
                        student_level=student_level,
                        difficulty=difficulty,
                        practice_type=practice_type,
                        num_questions=int(num_questions),
                        context_chunks=context_chunks,
                    )
                st.markdown(response["output"])

    with tabs[5]:
        st.subheader("Learning History Dashboard")
        history = load_history()
        metrics = get_history_metrics()

        if not history:
            st.info("No learning history logged yet. Complete a quiz to view your dashboard analytics!")
        else:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Quizzes Attempted", metrics["total_quizzes"])
            with col2:
                st.metric("Average Score", f"{metrics['average_score']}%")
            with col3:
                st.metric("Weakest Concept Focus", metrics["weakest_concept"])

            st.divider()

            df_history = pd.DataFrame([
                {"Attempt": f"#{idx+1} ({r['subject']})", "Score": r["score"]}
                for idx, r in enumerate(history)
            ])
            st.markdown("#### Performance Trend Chart")
            st.line_chart(df_history, x="Attempt", y="Score", color="#a855f7")

            st.divider()
            st.markdown("#### Past Quiz Logs")
            for idx, r in enumerate(reversed(history)):
                with st.expander(f"Attempt: {r['timestamp']} | Subject: {r['subject']} | Score: {r['score']}%"):
                    st.write(r["summary"])
                    st.markdown("**Concept Breakdowns:**")
                    for c in r.get("concepts", []):
                        status = c.get("status", "Moderate")
                        status_icon = "🟢" if status == "Strong" else "🟡" if status == "Moderate" else "🔴"
                        st.markdown(f"{status_icon} **{c['name']} ({c['score']}%)**: {c['feedback']}")


if __name__ == "__main__":
    main()
