"""
pipeline.py
-----------
RAG pipeline using the FREE Groq API (llama-3.1-8b-instant).

Speed design:
  - Groq runs at ~500 tokens/sec — fastest free LLM API available
  - Yields tokens as a generator so Streamlit streams them instantly
  - @st.cache_resource (applied in app.py) keeps Groq client alive
"""

import os
from groq import Groq
from langchain.schema import Document

GROQ_MODEL  = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
MAX_TOKENS  = 1024
TEMPERATURE = 0.0   # deterministic for factual Q&A


POLICY_SYSTEM = """You are a knowledgeable and friendly AI assistant helping
international students understand education loan policies.

Rules:
1. Answer ONLY from the CONTEXT provided. If the answer is not in the context,
   say: "I don't have enough information on that. Please check directly with
   the lender or a financial advisor."
2. Always cite your sources at the end: "Sources: [name]"
3. Keep answers clear and student-friendly. Avoid jargon.
4. Never fabricate interest rates, fees, or loan terms.
5. Remind students that loan terms change — always verify with the lender.
6. You provide information only — not legal or financial advice.

CONTEXT:
{context}
"""

CONTRACT_SYSTEM = """You are a helpful AI assistant explaining loan agreement
documents in plain English for international students.

Rules:
1. Answer ONLY from the DOCUMENT CONTEXT below.
2. Explain legal and financial terms simply.
3. Highlight key numbers: interest rates, fees, deadlines, repayment amounts.
4. Flag any clause the student should pay close attention to.
5. Reference which section your answer comes from.
6. If it's not in the document, say so clearly.
7. Remind the student to seek professional advice for major decisions.

DOCUMENT CONTEXT:
{context}
"""


def build_groq_client() -> Groq:
    """Create the Groq client. Cache this in app.py with @st.cache_resource."""
    return Groq(api_key=os.getenv("GROQ_API_KEY"))


def _format_context(docs: list) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown")
        lender = doc.metadata.get("lender_name", "")
        label  = f"[{i}] {source}" + (f" ({lender})" if lender else "")
        parts.append(f"{label}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def _build_messages(query: str, context: str, history: list, mode: str) -> list:
    """Build the Groq messages list with system prompt + last 4 chat turns."""
    template  = POLICY_SYSTEM if mode == "policy" else CONTRACT_SYSTEM
    messages  = [{"role": "system", "content": template.format(context=context)}]

    for turn in history[-8:]:   # last 4 turns = 8 messages
        messages.append({"role": turn["role"], "content": turn["content"]})

    messages.append({"role": "user", "content": query})
    return messages


def stream_answer(query: str, retriever, client: Groq,
                  history: list = None, mode: str = "policy"):
    """
    Retrieve relevant chunks and stream the LLM answer.

    Returns:
        (token_generator, source_docs)
        token_generator yields str tokens for st.write_stream()
        source_docs is a list of LangChain Document objects
    """
    if history is None:
        history = []

    docs     = retriever.invoke(query)
    context  = _format_context(docs)
    messages = _build_messages(query, context, history, mode)

    stream = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        stream=True,
    )

    def token_gen():
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    return token_gen(), docs


def get_source_summary(docs: list) -> list:
    """Return a clean list of source dicts for UI display."""
    seen, sources = set(), []
    for doc in docs:
        source = doc.metadata.get("source", "Unknown")
        if source not in seen:
            seen.add(source)
            snippet = doc.page_content[:300].replace("\n", " ").strip()
            sources.append({
                "label":   source,
                "snippet": snippet + "…" if len(doc.page_content) > 300 else snippet,
                "url":     doc.metadata.get("url", ""),
            })
    return sources
