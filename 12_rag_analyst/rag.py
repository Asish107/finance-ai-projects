"""
Project 12: RAG - ask questions about a company's real annual report (10-K)
----------------------------------------------------------------------------
Goal: Build a system where an AI answers questions about a SPECIFIC document it
has never seen, grounded in that document instead of its general memory. This
is RAG (Retrieval-Augmented Generation), the most in-demand AI skill today.

Why plain LLMs are not enough:
  If you ask an AI "what are Apple's biggest risks this year?", it answers from
  vague training memory, which may be outdated or made up. RAG fixes this by
  FEEDING the AI the actual, current document and telling it to answer only
  from what it was given.

The RAG recipe (this whole file):
  1. FETCH   - download the company's latest 10-K annual report from the SEC.
  2. CHUNK   - split the huge document into small paragraphs.
  3. RETRIEVE- for your question, find the few most relevant chunks
               (here with simple keyword matching / TF-IDF).
  4. GENERATE- hand ONLY those chunks + your question to the LLM to answer.

The magic is step 3+4: the AI never reads all 100 pages. It reads the ~5 most
relevant paragraphs we hand it. That is what makes RAG cheap, fast, and grounded.

Setup:  export OPENROUTER_API_KEY="sk-or-..."
Run:    python3 rag.py AAPL "What are the biggest risks the company faces?"
        python3 rag.py MSFT "How does the company make most of its money?"
        python3 rag.py NVDA "What does it say about competition?" --no-ai
"""

import os
import re
import sys
import json
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# The SEC requires a User-Agent that identifies you. Any real contact works.
SEC_HEADERS = {"User-Agent": "finance-ai-learning contact@example.com"}
MODEL = "openai/gpt-4o-mini"


def get_10k_text(ticker: str) -> str:
    """Download the latest 10-K annual report for a ticker from SEC EDGAR."""
    # 1) Map ticker -> CIK (the SEC's internal company id).
    tmap = requests.get("https://www.sec.gov/files/company_tickers.json",
                        headers=SEC_HEADERS, timeout=30).json()
    cik = None
    for row in tmap.values():
        if row["ticker"].upper() == ticker.upper():
            cik = str(row["cik_str"]).zfill(10)
            break
    if cik is None:
        raise ValueError(f"No SEC company found for ticker '{ticker}'.")

    # 2) Get the company's filing history and find the most recent 10-K.
    subs = requests.get(f"https://data.sec.gov/submissions/CIK{cik}.json",
                        headers=SEC_HEADERS, timeout=30).json()
    recent = subs["filings"]["recent"]
    idx = next((i for i, f in enumerate(recent["form"]) if f == "10-K"), None)
    if idx is None:
        raise ValueError(f"No 10-K filing found for {ticker}.")

    accession = recent["accessionNumber"][idx].replace("-", "")
    doc = recent["primaryDocument"][idx]
    cik_int = int(cik)
    url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession}/{doc}"

    # 3) Download the filing and strip the HTML down to plain text.
    html = requests.get(url, headers=SEC_HEADERS, timeout=60).text
    text = re.sub(r"<[^>]+>", " ", html)          # remove HTML tags
    text = re.sub(r"&#\d+;|&[a-z]+;", " ", text)  # remove HTML entities
    text = re.sub(r"\s+", " ", text)              # collapse whitespace
    return text


def chunk_text(text: str, words_per_chunk: int = 220):
    """Split the document into small, overlapping paragraphs for retrieval."""
    words = text.split()
    chunks = []
    step = words_per_chunk - 40   # 40-word overlap so ideas aren't cut in half
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + words_per_chunk])
        if len(chunk) > 200:      # skip tiny fragments
            chunks.append(chunk)
    return chunks


def retrieve(question: str, chunks, top_k: int = 5):
    """Find the chunks most relevant to the question (TF-IDF keyword matching).

    This is the 'R' in RAG. TF-IDF scores how well each chunk's words match the
    question. Real production systems use 'embeddings' (semantic search) here,
    which also catch synonyms, but the idea is identical: rank, then take the top few.
    """
    vec = TfidfVectorizer(stop_words="english")
    matrix = vec.fit_transform(chunks + [question])
    scores = cosine_similarity(matrix[-1], matrix[:-1])[0]
    best = scores.argsort()[::-1][:top_k]
    return [chunks[i] for i in best]


def run(ticker: str, question: str, use_ai: bool = True):
    print(f"\nFetching {ticker}'s latest 10-K annual report from the SEC...")
    text = get_10k_text(ticker)
    chunks = chunk_text(text)
    print(f"Report loaded: ~{len(text.split()):,} words, split into {len(chunks)} chunks.\n")

    print(f"Finding the paragraphs most relevant to: \"{question}\"\n")
    top_chunks = retrieve(question, chunks)

    print("=" * 66)
    print("  TOP RETRIEVED PASSAGES (what the AI will read)")
    print("=" * 66)
    for i, c in enumerate(top_chunks, 1):
        print(f"\n  [{i}] {c[:300]}...")
    print("\n" + "=" * 66)

    if not use_ai:
        print("\n  (--no-ai: showing retrieval only. Add your key for the AI answer.)\n")
        return

    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        print('\n  Set your key for the AI answer:  export OPENROUTER_API_KEY="sk-or-..."\n')
        return

    from openai import OpenAI
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)

    # This is the heart of RAG: we give the model ONLY the retrieved passages
    # and instruct it to answer strictly from them.
    context = "\n\n---\n\n".join(top_chunks)
    system = (
        "You are a financial analyst assistant. Answer the user's question using "
        "ONLY the provided excerpts from the company's official 10-K filing. "
        "If the excerpts don't contain the answer, say so honestly. Quote or cite "
        "specifics where possible. Be clear and concise. Not investment advice."
    )
    user = f"EXCERPTS FROM {ticker}'s 10-K:\n{context}\n\nQUESTION: {question}"

    print("\n  Asking the AI to answer from those passages...\n")
    resp = client.chat.completions.create(
        model=MODEL, temperature=0.2,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}])
    print("=" * 66)
    print(f"  AI ANSWER (grounded in {ticker}'s actual 10-K)")
    print("=" * 66 + "\n")
    print(resp.choices[0].message.content + "\n")

    print(
        "  What you built:\n"
        "  • The AI answered from THIS company's real filing, not vague memory.\n"
        "  • It only ever saw ~5 paragraphs, not the whole 100-page report,\n"
        "    which is why RAG is cheap and fast even on huge documents.\n"
        "  • Same pattern powers 'chat with your PDF / docs' tools everywhere.\n"
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python3 rag.py TICKER "your question"  [--no-ai]')
        sys.exit(0)
    ticker = sys.argv[1].upper()
    no_ai = "--no-ai" in sys.argv
    args = [a for a in sys.argv[2:] if a != "--no-ai"]
    question = " ".join(args) or "What are the biggest risks the company faces?"
    run(ticker, question, use_ai=not no_ai)
