"""Hybrid retrieval: BM25 (keywords) + embeddings (meaning), merged with RRF."""
import hashlib
import re
import sys
from pathlib import Path
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import snowballstemmer

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from app.chunker import EMBED_MODEL, build_chunks, load_documents

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / ".cache"
RRF_K = 60          # standard Reciprocal Rank Fusion constant

RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
USE_RERANKER = os.getenv("USE_RERANKER", "true").lower() == "true"


_stemmer = snowballstemmer.stemmer("english")

def tokenize(text):
    """Lowercase, split into words, reduce each word to its stem (villas -> villa)."""
    return _stemmer.stemWords(re.findall(r"\w+", text.lower()))


class HybridRetriever:
    def __init__(self):
        self.chunks = build_chunks(load_documents())
        texts = [c["text"] for c in self.chunks]
        self.bm25 = BM25Okapi([tokenize(t) for t in texts])
        self.model = SentenceTransformer(EMBED_MODEL)
        self.vectors = self._load_or_embed(texts)
        self.reranker = CrossEncoder(RERANK_MODEL, max_length=256) if USE_RERANKER else None

    def _load_or_embed(self, texts):
        """Embed all chunks once and cache to disk; re-embed only if the data changes."""
        key = hashlib.sha256("\n".join(texts).encode("utf-8")).hexdigest()[:16]
        path = CACHE_DIR / f"embeddings_{key}.npy"
        if path.exists():
            return np.load(path)
        print(f"Embedding {len(texts)} chunks (only when data changes)...")
        vectors = self.model.encode(texts, normalize_embeddings=True,
                                    batch_size=64, show_progress_bar=True)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.save(path, vectors)
        return vectors

    def bm25_ranking(self, query):
        scores = self.bm25.get_scores(tokenize(query))
        order = np.argsort(-scores)
        return [i for i in order if scores[i] > 0]          # only chunks sharing a word

    def vector_ranking(self, query):
        q = self.model.encode([query], normalize_embeddings=True)[0]
        return list(np.argsort(-(self.vectors @ q)))          # cosine similarity

    def search(self, query, k=30, depth=100, max_per_doc=2, max_overlap=0.8):
        """Hybrid search: BM25 + vectors merged with RRF, then de-duplicated."""
        fused = {}
        for ranking in (self.bm25_ranking(query), self.vector_ranking(query)):
            for rank, idx in enumerate(ranking[:depth]):
                fused[int(idx)] = fused.get(int(idx), 0.0) + 1.0 / (RRF_K + rank + 1)

        results, per_doc, picked_words = [], {}, []
        for i in sorted(fused, key=fused.get, reverse=True):
            chunk = self.chunks[i]
            if per_doc.get(chunk["doc_id"], 0) >= max_per_doc:
                continue
            words = set(tokenize(chunk["text"].split("\n", 1)[-1]))   # body, without header
            if len(words) >= 8 and any(
                    len(words & w) / max(min(len(words), len(w)), 1) > max_overlap
                    for w in picked_words):
                continue                                              # near-duplicate text
            per_doc[chunk["doc_id"]] = per_doc.get(chunk["doc_id"], 0) + 1
            picked_words.append(words)
            results.append({**chunk, "score": round(fused[i], 4), "hybrid_rank": len(results) + 1})
            if len(results) == k:
                break
        return results

    def rerank(self, query, candidates, k=6):
        """Score each (question, chunk) pair with the cross-encoder and keep the best k."""
        if self.reranker is None or not candidates:
            return candidates[:k]
        scores = self.reranker.predict([(query, c["text"]) for c in candidates])
        for c, s in zip(candidates, scores):
            c["rerank_score"] = round(float(s), 3)
        return sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)[:k]

    def retrieve(self, query, k=6, candidates=30):
        """Full pipeline: hybrid search for candidates, then rerank to the final k."""
        return self.rerank(query, self.search(query, k=candidates), k=k)

if __name__ == "__main__":
    retriever = HybridRetriever()
    query = " ".join(sys.argv[1:]) or "When will Trump Tower Jeddah be completed?"
    print(f"\nQuery: {query}\n")
    for hit in retriever.retrieve(query, k=5):
        print(f"[rerank {hit.get('rerank_score')} | was hybrid #{hit['hybrid_rank']}] "
              f"{hit['text'][:200]}\n")