"""Split scraped records into small, self-contained chunks for retrieval.

Sizes are measured in real tokens with the embedding model's own tokenizer,
so no chunk (header + body) exceeds the model's input limit.
"""
import json
import re
from pathlib import Path

from transformers import AutoTokenizer
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MODEL_MAX_TOKENS = 128          # model's input limit
SPECIAL_TOKENS = 2              # [CLS] and [SEP] added by the model
META_KEYS = {"source", "type", "title", "url", "project"}
SENTENCE_END = re.compile(r"(?<=[.!?؟])\s+")
FACT_MAX_CHARS = 200            # values up to this length are facts

tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)


def n_tokens(text):
    return len(tokenizer.encode(text, add_special_tokens=False))


def load_documents():
    docs = []
    for f in sorted(DATA_DIR.glob("*.json")):
        docs += json.loads(f.read_text(encoding="utf-8"))
    return docs


def fmt(value):
    if isinstance(value, float):
        return f"{int(value):,}" if value.is_integer() else str(value)
    return str(value)


def cut_long_sentence(sentence, budget):
    """Split one over-long sentence into word groups that fit the budget."""
    parts, current = [], []
    for word in sentence.split():
        if current and n_tokens(" ".join(current + [word])) > budget:
            parts.append(" ".join(current))
            current = []
        current.append(word)
    if current:
        parts.append(" ".join(current))
    return parts


def split_text(text, budget):
    """Split at sentence ends into chunks of <= budget tokens, 1-sentence overlap."""
    pieces = []
    for sentence in SENTENCE_END.split(text):
        sentence = sentence.strip()
        if not sentence:
            continue
        pieces += cut_long_sentence(sentence, budget) if n_tokens(sentence) > budget else [sentence]

    chunks, current = [], []
    for piece in pieces:
        if current and n_tokens(" ".join(current + [piece])) > budget:
            chunks.append(" ".join(current))
            last = current[-1]
            # carry the last sentence over only if it leaves room for new text
            current = [last] if n_tokens(last) < budget // 2 else []
            if current and n_tokens(" ".join(current + [piece])) > budget:
                current = []
        current.append(piece)
    if current:
        chunks.append(" ".join(current))
    return chunks


def split_facts(facts, budget):
    """Group 'key: value' facts into chunks without cutting any fact in half."""
    chunks, current = [], ""
    for key, value in facts:
        item = f"{key.replace('_', ' ')}: {value}"
        candidate = f"{current}; {item}" if current else item
        if current and n_tokens(candidate) > budget:
            chunks.append(current)
            current = item
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def header(doc):
    """One line that tells the model (and the search) what this chunk belongs to."""
    if doc["type"] == "faq":
        about = doc.get("project") or doc.get("city") or ""
        return f"{doc['source']} FAQ ({about}): {doc['title']}"
    location = doc.get("location") or doc.get("city") or ""
    return f"{doc['source']} {doc['type']}: {doc['title']}" + (f" | {location}" if location else "")


def doc_to_chunks(doc, doc_id):
    head = header(doc)
    # tokens left for the body after the header, the newline and special tokens
    budget = max(MODEL_MAX_TOKENS - SPECIAL_TOKENS - n_tokens(head + "\n"), 32)

    facts, paragraphs = [], []
    for key, value in doc.items():
        if key in META_KEYS or value in (None, "", []):
            continue
        text = fmt(value)
        if key == "answer" or len(text) > FACT_MAX_CHARS:
            paragraphs.append((key, text))
        else:
            facts.append((key, text))

    bodies = []
    if facts and doc["type"] != "faq":
        bodies += split_facts(facts, budget)
    for key, text in paragraphs:
        label = "" if key in ("answer", "description", "page_text") else key.replace("_", " ") + ": "
        bodies += [label + part for part in split_text(text, budget - n_tokens(label))]
    if not bodies:
        bodies = [""]

    return [{
        "doc_id": doc_id,
        "source": doc["source"],
        "type": doc["type"],
        "title": doc.get("title"),
        "url": doc.get("url"),
        "text": f"{head}\n{body}".strip(),
    } for body in bodies]


def build_chunks(docs):
    chunks = []
    for i, doc in enumerate(docs):
        chunks += doc_to_chunks(doc, i)
    return chunks


if __name__ == "__main__":
    from collections import Counter
    docs = load_documents()
    chunks = build_chunks(docs)
    tok = sorted(n_tokens(c["text"]) + SPECIAL_TOKENS for c in chunks)
    print(f"{len(docs)} documents -> {len(chunks)} chunks")
    print("chunks by source/type:", Counter((c["source"], c["type"]) for c in chunks))
    print("tokens per chunk min/median/max:", tok[0], tok[len(tok) // 2], tok[-1],
          f"(limit {MODEL_MAX_TOKENS})")
    print("chunks over limit:", sum(1 for t in tok if t > MODEL_MAX_TOKENS))
    print("\n---- sample: first chunks of one DarGlobal project ----")
    for c in [c for c in chunks if c["title"] == "Trump Tower Jeddah"][:3]:
        print(c["text"], "\n")