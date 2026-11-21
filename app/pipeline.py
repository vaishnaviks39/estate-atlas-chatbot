"""RAG pipeline: understand the question, retrieve context + matching listings, get the answer."""
import sys

from app.chunker import load_documents
from app.listing_filter import ListingFilter, format_listing
from app.llm import chat, stream_chat
from app.query import analyze
from app.rag import HybridRetriever

SYSTEM_PROMPT = """You are a real-estate assistant for two sources:
- DarGlobal (darglobal.co.uk): a luxury developer of branded residences in the UAE, Saudi Arabia, Qatar, Oman, Spain and the UK.
- Wasalt (wasalt.sa): a Saudi property portal with listings for sale and rent.

Rules:
1. Answer ONLY from the numbered CONTEXT, MATCHING LISTINGS and DARGLOBAL PROJECT CATALOG below. If the answer is not there, say you don't have that information in the scraped data. Never guess.
2. When MATCHING LISTINGS are given, they exactly match the user's filters: present them first.
3. For questions that list, count or compare DarGlobal projects (e.g. "which projects are in Oman?"), use the full DARGLOBAL PROJECT CATALOG so the answer is complete.
4. Never invent prices, dates, sizes or availability. DarGlobal does not publish prices.
5. Cite sources inline with their numbers, like [1] or [2][3].
6. Be concise. Use short bullet lists when listing several properties. Write prices like "SAR 1,450,000".
7. Wasalt rent listings do not state whether rent is monthly or yearly; mention this when giving rents.
8. Reply in the same language as the user's question (English or Arabic).
"""
ORDER_LABEL = {"newest": "newest", "price_low": "cheapest", "price_high": "most expensive"}


class RAGPipeline:
    def __init__(self):
        docs = load_documents()
        self.retriever = HybridRetriever()     # chunks, embeddings, reranker (loaded once)
        self.listing_filter = ListingFilter(docs)
        self.projects = [d for d in docs if d.get("source") == "DarGlobal" and d.get("type") == "project"]

    def prepare(self, question, history=None, query=None):
        """Understand the question, gather context, build the messages for the model."""
        query = query or analyze(question, history)
        wants_listings = query["wants_listings"]

        # 1) text retrieval (FAQs, projects, guides...) using the rewritten English query
        hits = self.retriever.retrieve(query["search_query"], k=4 if wants_listings else 6)
        sources = [{"n": i, "title": h["title"], "url": h["url"], "source": h["source"]}
                   for i, h in enumerate(hits, 1)]
        context = "\n\n".join(f"[{s['n']}] {h['text']}\nSource: {h['url']}"
                              for s, h in zip(sources, hits))

        # 2) metadata filtering on Wasalt listings, if the user is looking for properties
        listing_block = ""
        if wants_listings:
            found, total = self.listing_filter.search(query)
            order = ORDER_LABEL.get(query.get("sort"), "newest")
            filters = {k: v for k, v in query.items()
                       if k not in ("search_query", "wants_listings", "sort") and v is not None}
            if found:
                lines = []
                for d in found:
                    n = len(sources) + 1
                    sources.append({"n": n, "title": d["title"], "url": d["url"], "source": "Wasalt"})
                    lines.append(f"[{n}] {format_listing(d)}\nSource: {d['url']}")
                listing_block = (f"\n\nMATCHING LISTINGS (filters: {filters}; {total} unique matches in "
                                 f"our scraped Wasalt sample, showing the {len(found)} {order}):\n"
                                 + "\n".join(lines))
            else:
                listing_block = (f"\n\nMATCHING LISTINGS: none. No Wasalt listings in the scraped data "
                                 f"match filters {filters}. Say so and suggest relaxing a filter.")

        # 3) compact catalog of ALL DarGlobal projects, so list/count questions are complete
        catalog_lines = []
        for d in self.projects:
            n = len(sources) + 1
            sources.append({"n": n, "title": d["title"], "url": d["url"], "source": "DarGlobal"})
            facts = " | ".join(str(d[k]) for k in ("location", "property_type", "completion") if d.get(k))
            catalog_lines.append(f"[{n}] {d['title']} | {facts}")
        catalog_block = (f"\n\nDARGLOBAL PROJECT CATALOG (all {len(self.projects)} projects):\n"
                         + "\n".join(catalog_lines))

        system = SYSTEM_PROMPT + "\nCONTEXT:\n" + context + listing_block + catalog_block
        messages = [{"role": "system", "content": system}]
        messages += (history or [])[-6:]
        messages.append({"role": "user", "content": question})
        return messages, sources, query

    def answer(self, question, history=None):
        """Full answer at once (used for testing)."""
        messages, sources, query = self.prepare(question, history)
        return chat(messages), sources, query

    def stream_answer(self, question, history=None):
        """Answer piece by piece (used by the API for live typing)."""
        messages, sources, query = self.prepare(question, history)
        return stream_chat(messages), sources, query


if __name__ == "__main__":
    pipeline = RAGPipeline()
    question = " ".join(sys.argv[1:]) or "villas for sale in Riyadh under 3 million"
    answer, sources, query = pipeline.answer(question)
    print(f"\nQ: {question}\nUnderstood as: {query}\n\nA: {answer}\n\nSources:")
    for s in sources:
        print(f"  [{s['n']}] {s['source']} | {s['title']} | {s['url']}")