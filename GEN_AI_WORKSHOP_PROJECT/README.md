# GenAI Learning Mentor

Streamlit app that builds a FAISS vector store from uploaded PDFs and uses Google Gemini + LangChain RAG to:
- Answer questions from uploaded notes
- Generate personalized study plans
- Generate quizzes
- Identify weak areas
- Generate practice questions

## Setup

1. Create a virtual environment (recommended)
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set `GOOGLE_API_KEY`.

Copy `.env.example` to `.env` and fill in your key.

## Run

```bash
streamlit run app.py
```


