"""Ranking papers for a query: semantic (embedding) search and keyword (BM25) search.

The Retriever agent gets candidate papers from Semantic Scholar and arXiv and
uses a ranker to pick the abstracts most relevant to each sub-question.
scripts/eval_retrieval.py compares the two rankers on SciFact.

Both rankers work the same way:
    ranker = make_ranker("semantic")          # or "bm25"
    ranker.fit(list_of_abstracts)
    hits = ranker.search(["query 1", "query 2"], k=5)
    # hits[i] = [(doc_index, score), ...], best first, for query i
"""

from __future__ import annotations

import re

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
# BGE v1.5 models expect this prefix on short queries that search longer passages.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

Hits = list[list[tuple[int, float]]]


def tokenize(text: str) -> list[str]:
    """Lowercase words and numbers, without common English stop words."""
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in ENGLISH_STOP_WORDS]


def _top_k(scores: np.ndarray, k: int) -> list[tuple[int, float]]:
    k = min(k, len(scores))
    idx = np.argpartition(-scores, k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return [(int(i), float(scores[i])) for i in idx]


class BM25Ranker:
    """Keyword search (Okapi BM25): the baseline the proposal compares against."""

    name = "bm25"

    def __init__(self):
        self._bm25: BM25Okapi | None = None

    def fit(self, docs: list[str]) -> "BM25Ranker":
        self._bm25 = BM25Okapi([tokenize(d) or ["_"] for d in docs])
        return self

    def search(self, queries: list[str], k: int) -> Hits:
        if self._bm25 is None:
            raise RuntimeError("call fit() first")
        return [_top_k(np.asarray(self._bm25.get_scores(tokenize(q))), k) for q in queries]


class SemanticRanker:
    """Embedding search: cosine similarity between query and abstract embeddings."""

    name = "semantic"

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL, device: str | None = None, batch_size: int = 32):
        from sentence_transformers import SentenceTransformer  # slow import; only when used

        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size
        self.query_prefix = BGE_QUERY_PREFIX if "bge-" in model_name.lower() else ""
        self._doc_emb: np.ndarray | None = None

    def encode(self, texts: list[str], is_query: bool, show_progress: bool = False) -> np.ndarray:
        prompt = self.query_prefix if is_query and self.query_prefix else None
        return self.model.encode(
            texts,
            prompt=prompt,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
        )

    def fit(self, docs: list[str], show_progress: bool = False) -> "SemanticRanker":
        self._doc_emb = self.encode(docs, is_query=False, show_progress=show_progress)
        return self

    def search(self, queries: list[str], k: int) -> Hits:
        if self._doc_emb is None:
            raise RuntimeError("call fit() first")
        scores = self.encode(queries, is_query=True) @ self._doc_emb.T
        return [_top_k(row, k) for row in scores]


def make_ranker(method: str, **kwargs) -> BM25Ranker | SemanticRanker:
    if method == "bm25":
        return BM25Ranker()
    if method == "semantic":
        return SemanticRanker(**kwargs)
    raise ValueError(f"unknown retrieval method {method!r}; use 'semantic' or 'bm25'")
