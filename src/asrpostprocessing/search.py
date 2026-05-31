from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import List

from .config import ExperimentConfig
from .schemas import SearchResult


class CachedSearchProvider:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.cache_dir = Path(config.search_cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def search(self, query: str) -> List[SearchResult]:
        if not self.config.enable_search:
            return []
        cache_path = self._cache_path(query)
        if cache_path.exists():
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            return [SearchResult(**item) for item in payload.get("results", [])]
        results = self._fetch(query)
        payload = {"query": query, "created_at": time.time(), "results": [result.to_dict() for result in results]}
        cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return results

    def _fetch(self, query: str) -> List[SearchResult]:
        provider = (self.config.search_provider or "").lower()
        if provider in {"duckduckgo", "ddg", "duckduckgo_instant"}:
            return self._fetch_duckduckgo(query)
        if provider in {"none", "off", "disabled"}:
            return []
        if not self.config.search_endpoint:
            return []
        return self._fetch_endpoint(query)

    def _fetch_endpoint(self, query: str) -> List[SearchResult]:
        try:
            import requests  # type: ignore

            response = requests.get(
                self.config.search_endpoint,
                params={"q": query, "strength": self.config.search_strength},
                timeout=self.config.request_timeout_s,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception:
            return []
        results: List[SearchResult] = []
        for item in payload.get("results", [])[:5]:
            results.append(
                SearchResult(
                    query=query,
                    title=str(item.get("title", "")),
                    url=str(item.get("url", "")),
                    snippet=str(item.get("snippet", "")),
                    source="remote",
                )
            )
        return results

    def _fetch_duckduckgo(self, query: str) -> List[SearchResult]:
        try:
            import requests  # type: ignore

            response = requests.get(
                "https://api.duckduckgo.com/",
                params={"q": query, "format": "json", "no_redirect": "1", "no_html": "1", "skip_disambig": "1"},
                timeout=min(self.config.request_timeout_s, 10),
            )
            response.raise_for_status()
            payload = response.json()
        except Exception:
            return []
        results: List[SearchResult] = []
        if payload.get("AbstractText"):
            results.append(
                SearchResult(
                    query=query,
                    title=str(payload.get("Heading") or query),
                    url=str(payload.get("AbstractURL") or ""),
                    snippet=str(payload.get("AbstractText") or ""),
                    source="duckduckgo",
                )
            )
        for item in _flatten_related_topics(payload.get("RelatedTopics", [])):
            text = str(item.get("Text") or "")
            if not text:
                continue
            results.append(
                SearchResult(
                    query=query,
                    title=text.split(" - ", 1)[0][:120],
                    url=str(item.get("FirstURL") or ""),
                    snippet=text,
                    source="duckduckgo",
                )
            )
            if len(results) >= 5:
                break
        return results

    def _cache_path(self, query: str) -> Path:
        key = f"{self.config.search_provider}:{query}"
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
        return self.cache_dir / f"{digest}.json"


def _flatten_related_topics(items):
    for item in items or []:
        if isinstance(item, dict) and "Topics" in item:
            yield from _flatten_related_topics(item.get("Topics", []))
        elif isinstance(item, dict):
            yield item
