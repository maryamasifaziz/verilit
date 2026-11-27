"""Paper search: Semantic Scholar and arXiv.

Both return papers with title and abstract. Semantic Scholar is searched
first; arXiv adds preprints. Requests to Semantic Scholar are spaced at least
1.1 s apart (its limit is 1 per second) and arXiv requests 3 s apart (as its
API terms ask), with exponential backoff on 429 / 5xx errors. Every result is
cached on disk.

Set S2_API_KEY to use your Semantic Scholar key. Without one, requests share a
public rate limit and often get 429; after a few retries the search moves on
with arXiv results only.
"""

from __future__ import annotations

import os
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field

import httpx
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from verilit.cache import MISSING, DiskCache

S2_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
S2_FIELDS = "paperId,title,abstract,year,authors,externalIds,url,venue"
ARXIV_URL = "https://export.arxiv.org/api/query"
ATOM = {"a": "http://www.w3.org/2005/Atom"}


@dataclass
class Paper:
    paper_id: str  # "s2:<id>" or "arxiv:<id>"
    title: str
    abstract: str
    year: int | None = None
    authors: list[str] = field(default_factory=list)
    url: str | None = None
    venue: str | None = None
    source: str = ""

    def text(self) -> str:
        """Title + abstract: what the ranker and the Verifier read (same format as SciFact)."""
        return f"{self.title.strip()} {self.abstract.strip()}"

    def citation(self) -> str:
        names = self.authors[:3]
        who = ", ".join(names) + (" et al." if len(self.authors) > 3 else "") if names else "Unknown authors"
        year = self.year or "n.d."
        venue = f" {self.venue}." if self.venue else ""
        url = f" {self.url}" if self.url else ""
        return f"{who} ({year}). {self.title.rstrip('.')}.{venue}{url}"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Paper":
        return cls(**d)


def _clean(text: str | None) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _norm_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]", "", title.lower())


class PaperSearch:
    def __init__(self, use_s2: bool = True, use_arxiv: bool = True, use_cache: bool = True, verbose: bool = True):
        self.use_s2, self.use_arxiv = use_s2, use_arxiv
        self.cache = DiskCache("search") if use_cache else None
        self.verbose = verbose
        self.s2_key = os.environ.get("S2_API_KEY")
        self.http = httpx.Client(timeout=30, headers={"User-Agent": "VeriLit/0.1 (academic course project)"})
        self._last = {"s2": 0.0, "arxiv": 0.0}
        self.requests = {"s2": 0, "arxiv": 0}
        # After Semantic Scholar keeps refusing (429), skip it for a while instead of
        # spending a minute of retries on every query.
        self._s2_paused_until = 0.0
        self._last_failure_status: int | None = None
        self._s2_failures_in_a_row = 0

    def search(self, query: str, s2_limit: int = 20, arxiv_limit: int = 10) -> list[Paper]:
        """Papers for a query from both sources, without duplicates or papers lacking an abstract."""
        papers: list[Paper] = []
        if self.use_s2 and time.time() >= self._s2_paused_until:
            papers += self._cached("s2", query, s2_limit, self._search_s2)
        if self.use_arxiv:
            papers += self._cached("arxiv", query, arxiv_limit, self._search_arxiv)
        seen, unique = set(), []
        for p in papers:
            key = _norm_title(p.title)
            if p.abstract and len(p.abstract) > 100 and key not in seen:
                seen.add(key)
                unique.append(p)
        return unique

    # -- internals -------------------------------------------------------

    def _cached(self, source: str, query: str, limit: int, fn) -> list[Paper]:
        key = {"source": source, "query": query, "limit": limit}
        if self.cache is not None:
            cached = self.cache.get(key)
            if cached is not MISSING:
                return [Paper.from_dict(d) for d in cached]
        papers = fn(query, limit)
        if papers is None:  # failed after retries: don't cache, so a later run tries again
            return []
        if self.cache is not None:
            self.cache.set(key, [p.to_dict() for p in papers])
        return papers

    def _get(self, source: str, url: str, params: dict, headers: dict | None, min_interval: float, attempts: int = 5):
        """GET with spacing and backoff. Returns the response, or None after the last failed attempt."""
        status, detail = None, ""
        for attempt in range(attempts):
            wait = self._last[source] + min_interval - time.time()
            if wait > 0:
                time.sleep(wait)
            self._last[source] = time.time()
            self.requests[source] += 1
            try:
                r = self.http.get(url, params=params, headers=headers)
            except httpx.HTTPError as e:
                status, detail = None, str(e)
            else:
                if r.status_code == 200:
                    return r
                status, detail = r.status_code, r.text[:200]
                if status not in (429, 500, 502, 503, 504):
                    break
            delay = min(30, 3 * 2**attempt)
            if self.verbose:
                print(f"  [{source}] {status or 'error'}; retrying in {delay}s ({attempt + 1}/{attempts})")
            time.sleep(delay)
        if self.verbose:
            print(f"  [{source}] giving up on this query ({status}: {detail[:80]})")
        self._last_failure_status = status
        return None

    def _search_s2(self, query: str, limit: int) -> list[Paper] | None:
        headers = {"x-api-key": self.s2_key} if self.s2_key else None
        # Semantic Scholar often answers 429 even well under its limit, so space requests
        # out and retry patiently; pause it only after two queries in a row fail.
        r = self._get("s2", S2_URL, {"query": query, "limit": limit, "fields": S2_FIELDS}, headers, 1.5, 4)
        if r is None:
            self._s2_failures_in_a_row += 1
            if self._last_failure_status == 429 and self._s2_failures_in_a_row >= 2:
                self._s2_paused_until = time.time() + 600
                self._s2_failures_in_a_row = 0
                if self.verbose:
                    print("  [s2] rate-limited twice in a row; skipping Semantic Scholar for 10 minutes (arXiv only)."
                          + ("" if self.s2_key else " Set S2_API_KEY to avoid this."))
            return None
        self._s2_failures_in_a_row = 0
        papers = []
        for d in r.json().get("data") or []:
            ext = d.get("externalIds") or {}
            url = f"https://doi.org/{ext['DOI']}" if ext.get("DOI") else d.get("url")
            papers.append(
                Paper(
                    paper_id=f"s2:{d['paperId']}",
                    title=_clean(d.get("title")),
                    abstract=_clean(d.get("abstract")),
                    year=d.get("year"),
                    authors=[a.get("name", "") for a in d.get("authors") or []],
                    url=url,
                    venue=_clean(d.get("venue")) or None,
                    source="semantic_scholar",
                )
            )
        return papers

    def _search_arxiv(self, query: str, limit: int) -> list[Paper] | None:
        terms = [t for t in re.findall(r"[a-z0-9]+", query.lower()) if t not in ENGLISH_STOP_WORDS][:6]
        if not terms:
            return []
        papers = self._arxiv_request(" AND ".join(f"all:{t}" for t in terms), limit)
        if papers is not None and len(papers) < 3 and len(terms) > 2:
            # Requiring every word can be too strict; fall back to the first three words.
            papers = self._arxiv_request(" AND ".join(f"all:{t}" for t in terms[:3]), limit)
        return papers

    def _arxiv_request(self, search_query: str, limit: int) -> list[Paper] | None:
        params = {"search_query": search_query, "start": 0, "max_results": limit, "sortBy": "relevance"}
        r = self._get("arxiv", ARXIV_URL, params, None, 3.0)
        if r is None:
            return None
        root = ET.fromstring(r.content)
        papers = []
        for e in root.findall("a:entry", ATOM):
            abs_url = _clean(e.findtext("a:id", default="", namespaces=ATOM))
            arxiv_id = re.sub(r"v\d+$", "", abs_url.rsplit("/abs/", 1)[-1])
            published = e.findtext("a:published", default="", namespaces=ATOM)
            papers.append(
                Paper(
                    paper_id=f"arxiv:{arxiv_id}",
                    title=_clean(e.findtext("a:title", default="", namespaces=ATOM)),
                    abstract=_clean(e.findtext("a:summary", default="", namespaces=ATOM)),
                    year=int(published[:4]) if published[:4].isdigit() else None,
                    authors=[_clean(a.findtext("a:name", default="", namespaces=ATOM)) for a in e.findall("a:author", ATOM)],
                    url=f"https://arxiv.org/abs/{arxiv_id}",
                    venue="arXiv",
                    source="arxiv",
                )
            )
        return papers
