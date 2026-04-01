# 🎓 International Student Loan Assistant

A RAG-powered AI assistant — **100% free to run**.

---

## Free Tech Stack

| Component | Tool | Cost |
|---|---|---|
| LLM | Groq API — llama-3.1-8b-instant | Free (14,400 req/day) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 | Free (runs locally) |
| Vector DB | Pinecone free tier | Free (1 index) |
| UI | Streamlit | Free |

---

## Setup in 4 Steps

### Step 1 — Create virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### Step 2 — Install dependencies
```bash
pip install -r requirements.txt
```
> The first time you run the app, the MiniLM model (~90 MB) downloads once
> to ~/.cache/huggingface. Every run after that is instant.

### Step 3 — Set your API keys
```bash
cp .env.example .env
```
Open `.env` and fill in:
```
GROQ_API_KEY=gsk_...         # free at https://console.groq.com
PINECONE_API_KEY=pcsk_...    # free at https://app.pinecone.io
PINECONE_INDEX_NAME=loan-assistant
```

### Step 4 — Add corpus files and ingest
Copy the 6 scraped `.txt` files into `data/corpus/`, then run:
```bash
python -m rag.ingest
```
This creates the Pinecone index and uploads all vectors. Run once only.

### Step 5 — Start the app
```bash
streamlit run app.py
```
Opens at: http://localhost:8501

---

## Getting Free API Keys

**Groq (LLM):**
1. Go to https://console.groq.com
2. Sign up with Google / GitHub
3. Click "API Keys" → "Create API Key"
4. Copy key starting with `gsk_`

**Pinecone (Vector DB):**
1. Go to https://app.pinecone.io
2. Sign up free
3. Click "API Keys" → copy the default key
4. Free tier: 1 index, 2 GB storage — plenty for this project

---

## Project Structure
```
loan-assistant/
├── app.py                   Streamlit UI (Mode A + Mode B)
├── requirements.txt
├── .env.example
├── rag/
│   ├── ingest.py            One-time corpus indexing script
│   ├── retriever.py         Pinecone + MiniLM retriever
│   └── pipeline.py          Groq streaming RAG chain
├── utils/
│   ├── pdf_parser.py        PDF text extraction
│   └── chunker.py           Text splitting with metadata
└── data/
    └── corpus/              Add .txt policy files here
```

---

## Deploy Free on Streamlit Community Cloud

1. Push to GitHub
2. Go to https://share.streamlit.io → "New app"
3. Add secrets in the dashboard:
```toml
GROQ_API_KEY = "gsk_..."
PINECONE_API_KEY = "pcsk_..."
PINECONE_INDEX_NAME = "loan-assistant"
```
4. Deploy — no server costs, no credit card needed.
