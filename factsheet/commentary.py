from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import feedparser
from openai import OpenAI

DEFAULT_FEEDS: Sequence[str] = (
    "https://www.reuters.com/finance/markets/rss",
    "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
    "https://www.ft.com/markets?format=rss",
)

DEFAULT_MACRO_KEYWORDS: Sequence[str] = (
    "market",
    "markets",
    "equities",
    "equity",
    "stocks",
    "stock",
    "indexes",
    "indices",
    "nasdaq",
    "s&p",
    "dow",
    "bond",
    "bonds",
    "treasury",
    "yield",
    "yields",
    "inflation",
    "cpi",
    "ppi",
    "interest rate",
    "interest rates",
    "rate hike",
    "rate cut",
    "federal reserve",
    "fed",
    "central bank",
    "ecb",
    "bank of england",
    "boe",
    "bank of japan",
    "boj",
    "economy",
    "economic",
    "growth",
    "gdp",
    "recession",
    "geopolitical",
    "volatility",
    "commodity",
    "commodities",
    "oil",
    "energy",
    "gold",
    "silver",
    "currency",
    "currencies",
    "dollar",
    "euro",
    "yen",
    "outlook",
    "earnings",
)

DEFAULT_MAX_PER_FEED = 2
DEFAULT_MAX_SUMMARY_CHARS = 400


@dataclass
class FeedSnippet:
    source: str
    title: str
    summary: str
    published: str | None = None

    def to_prompt_fragment(self) -> str:
        date_part = f" ({self.published})" if self.published else ""
        return f"- {self.source}{date_part}: {self.title} ? {self.summary}"


def _clean_text(text: str) -> str:
    return " ".join(text.replace("\r", " ").replace("\n", " ").split())


def _coerce_int(value: Any, default: int | None = None) -> int | None:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalize_keywords(keywords: Sequence[str] | None) -> List[str]:
    normalized: List[str] = []
    if not keywords:
        return normalized
    for keyword in keywords:
        if not keyword:
            continue
        normalized.append(str(keyword).casefold())
    return normalized


def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    return [part.casefold() for part in re.split(r"[^A-Za-z0-9]+", text) if part]


def _context_holdings_keywords(context: Dict[str, Any]) -> List[str]:
    keywords: set[str] = set()
    holdings = context.get("holdings") or []
    for item in holdings:
        if not isinstance(item, dict):
            continue
        for key in ("name", "symbol", "yf_symbol"):
            value = item.get(key)
            if not value:
                continue
            value_str = str(value)
            keywords.add(value_str.casefold())
            for token in _tokenize(value_str):
                if len(token) >= 3:
                    keywords.add(token)
    for bucket in ("sectors", "regions"):
        entries = context.get(bucket) or []
        for entry in entries:
            if isinstance(entry, (list, tuple)) and entry:
                label = entry[0]
            else:
                label = entry
            if not label:
                continue
            label_str = str(label)
            keywords.add(label_str.casefold())
            for token in _tokenize(label_str):
                if len(token) >= 3:
                    keywords.add(token)
    return sorted(keywords)


def _snippet_text(snippet: FeedSnippet) -> str:
    return " ".join(part for part in (snippet.title, snippet.summary) if part).casefold()


def _contains_any(text: str, keywords: Sequence[str]) -> bool:
    return any(keyword and keyword in text for keyword in keywords)


def fetch_rss_snippets(
    feeds: Sequence[str] | None,
    max_per_feed: int = DEFAULT_MAX_PER_FEED,
    max_chars: int = DEFAULT_MAX_SUMMARY_CHARS,
) -> List[FeedSnippet]:
    snippets: List[FeedSnippet] = []
    feed_urls = feeds if feeds else DEFAULT_FEEDS
    for feed_url in feed_urls:
        try:
            parsed = feedparser.parse(feed_url)
        except Exception:
            continue
        entries = parsed.entries[:max_per_feed]
        source_name = parsed.feed.get("title", feed_url)
        for entry in entries:
            title = entry.get("title") or "Untitled"
            summary = entry.get("summary") or entry.get("description") or ""
            summary = _clean_text(summary)
            if len(summary) > max_chars:
                summary = summary[: max_chars - 3].rstrip() + "..."
            published = entry.get("published")
            snippets.append(
                FeedSnippet(source=source_name, title=title, summary=summary, published=published)
            )
    return snippets


def filter_snippets(
    snippets: Sequence[FeedSnippet],
    context: Dict[str, Any],
    config: Dict[str, Any],
) -> tuple[List[FeedSnippet], List[Dict[str, Any]]]:
    if not snippets:
        return [], []

    mode_value = config.get("mode", "all")
    mode = mode_value.lower() if isinstance(mode_value, str) else "all"
    if mode not in {"all", "macro", "holdings", "mixed"}:
        mode = "all"

    max_headlines = _coerce_int(config.get("max_headlines"))
    max_holdings = _coerce_int(config.get("max_holdings_headlines"), 2)
    max_macro = _coerce_int(config.get("max_macro_headlines"), 2)

    exclude_keywords = _normalize_keywords(config.get("exclude_keywords"))
    include_keywords = _normalize_keywords(config.get("include_keywords"))
    macro_keywords = _normalize_keywords(config.get("macro_keywords"))
    if not macro_keywords:
        macro_keywords = [kw.casefold() for kw in DEFAULT_MACRO_KEYWORDS]
    holdings_keywords = _context_holdings_keywords(context)
    if include_keywords:
        holdings_keywords = sorted(set(holdings_keywords) | set(include_keywords))

    records: List[Dict[str, Any]] = []
    for snippet in snippets:
        text = _snippet_text(snippet)
        record = {
            "snippet": snippet,
            "exclude": _contains_any(text, exclude_keywords),
            "include": _contains_any(text, include_keywords),
            "holdings": False,
            "macro": False,
        }
        record["holdings"] = record["include"] or _contains_any(text, holdings_keywords)
        record["macro"] = _contains_any(text, macro_keywords)
        records.append(record)

    non_excluded = [record for record in records if not record["exclude"]]
    if not non_excluded:
        return [], []

    selected_records: List[Dict[str, Any]]
    if mode == "all":
        selected_records = non_excluded
    elif mode == "macro":
        selected_records = [record for record in non_excluded if record["macro"] or record["include"]]
    elif mode == "holdings":
        selected_records = [record for record in non_excluded if record["holdings"] or record["include"]]
    else:  # mixed
        if max_holdings is not None and max_holdings < 0:
            max_holdings = 0
        if max_macro is not None and max_macro < 0:
            max_macro = 0

        selected_records = []
        remainder: List[Dict[str, Any]] = []
        holdings_count = 0
        macro_count = 0

        for record in non_excluded:
            if max_headlines is not None and len(selected_records) >= max_headlines:
                break
            if record["holdings"] and (max_holdings is None or holdings_count < max_holdings):
                selected_records.append(record)
                holdings_count += 1
            elif record["macro"] and (max_macro is None or macro_count < max_macro):
                selected_records.append(record)
                macro_count += 1
            else:
                remainder.append(record)

        if max_headlines is None or len(selected_records) < max_headlines:
            for record in remainder:
                if max_headlines is not None and len(selected_records) >= max_headlines:
                    break
                if record["holdings"] and (max_holdings is None or holdings_count < max_holdings):
                    selected_records.append(record)
                    holdings_count += 1
                elif record["macro"] and (max_macro is None or macro_count < max_macro):
                    selected_records.append(record)
                    macro_count += 1
                elif record["include"]:
                    selected_records.append(record)
                elif max_headlines is None or len(selected_records) < max_headlines:
                    selected_records.append(record)

    if mode in {"macro", "holdings"} and not selected_records:
        selected_records = non_excluded

    if not selected_records:
        selected_records = non_excluded

    if max_headlines is not None:
        selected_records = selected_records[:max_headlines]

    selected_snippets = [record["snippet"] for record in selected_records]
    debug_records = []
    for record in selected_records:
        snippet = record["snippet"]
        debug_records.append(
            {
                "source": snippet.source,
                "title": snippet.title,
                "published": snippet.published,
                "summary": snippet.summary,
                "matched_holdings": bool(record["holdings"]),
                "matched_macro": bool(record["macro"]),
                "matched_include_keywords": bool(record["include"]),
            }
        )

    return selected_snippets, debug_records


def build_prompt(context: Dict[str, Any], snippets: Sequence[FeedSnippet]) -> str:
    as_of = context.get("as_of")
    portfolio = context.get("portfolio", {})
    benchmark = context.get("benchmark", {})
    sectors = context.get("sectors", [])
    regions = context.get("regions", [])
    holdings = context.get("holdings", [])

    lines = [
        "You are crafting a concise managers commentary and outlook for an investment factsheet.",
        "Write in a professional, balanced tone, two to three short paragraphs.",
        "Mention key performance drivers, risk considerations, and forward-looking factors.",
        "Do not invent data; only use what is provided.",
        "If market context is provided, weave in the most relevant points.",
        "Avoid bullet points; respond in plain paragraphs.",
        "Factsheet Date: " + str(as_of),
        "",
        "Portfolio Performance:",
        f"- YTD Return: {portfolio.get('ytd')}",
        f"- Return Since Inception: {portfolio.get('since_inception')}",
        f"- Average Annual Compounded Return: {portfolio.get('cagr')}",
        "Benchmark Performance:",
        f"- YTD: {benchmark.get('ytd')}",
        f"- CAGR: {benchmark.get('cagr')}",
    ]

    if sectors:
        sector_lines = ", ".join(f"{name}: {value:.1%}" for name, value in sectors)
        lines.append("Sector Allocation: " + sector_lines)
    if regions:
        region_lines = ", ".join(f"{name}: {value:.1%}" for name, value in regions)
        lines.append("Regional Allocation: " + region_lines)
    if holdings:
        holdings_text = ", ".join(f"{item['name']} ({item['weight_pct']:.1f}%)" for item in holdings[:5])
        lines.append("Top Holdings: " + holdings_text)

    if snippets:
        lines.append("")
        lines.append("Recent Market Headlines:")
        for snippet in snippets:
            lines.append(snippet.to_prompt_fragment())

    lines.append("")
    lines.append("Compose the commentary now.")
    return "\n".join(lines)


def generate_commentary(context: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {
            "text": "Commentary generation disabled: OPENAI_API_KEY not set.",
            "error": "missing_api_key",
        }

    rss_feeds = config.get("rss_feeds")
    max_per_feed = _coerce_int(config.get("max_per_feed"), DEFAULT_MAX_PER_FEED)
    if max_per_feed is None or max_per_feed < 1:
        max_per_feed = DEFAULT_MAX_PER_FEED
    max_summary_chars = _coerce_int(config.get("max_summary_chars"), DEFAULT_MAX_SUMMARY_CHARS)
    if max_summary_chars is None or max_summary_chars < 80:
        max_summary_chars = DEFAULT_MAX_SUMMARY_CHARS

    try:
        fetched_snippets = fetch_rss_snippets(rss_feeds, max_per_feed=max_per_feed, max_chars=max_summary_chars)
    except Exception:
        fetched_snippets = []

    news_cache_path = config.get("news_cache")
    if news_cache_path:
        save_news_cache(fetched_snippets, news_cache_path)

    filtered_snippets: List[FeedSnippet]
    snippet_debug: List[Dict[str, Any]]
    try:
        filtered_snippets, snippet_debug = filter_snippets(fetched_snippets, context, config)
    except Exception:
        filtered_snippets = fetched_snippets
        snippet_debug = []

    prompt = build_prompt(context, filtered_snippets)
    client = OpenAI(api_key=api_key)
    model = config.get("model", "gpt-4o-mini")
    max_tokens = config.get("max_tokens", 700)
    temperature = config.get("temperature", 0.6)

    try:
        response = client.responses.create(
            model=model,
            input=prompt,
            max_output_tokens=max_tokens,
            temperature=temperature,
        )
        text = response.output_text.strip()
    except Exception as exc:  # pragma: no cover - network dependent
        return {
            "text": "Commentary generation failed; see logs for details.",
            "error": str(exc),
            "prompt": prompt,
        }

    return {
        "text": text,
        "prompt": prompt,
        "snippets": [snippet.__dict__ for snippet in filtered_snippets],
        "snippet_debug": snippet_debug,
        "fetched_snippets": [snippet.__dict__ for snippet in fetched_snippets],
    }


def save_debug_payload(payload: Dict[str, Any], path: str) -> None:
    try:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
    except OSError:
        pass


def save_news_cache(snippets: Sequence[FeedSnippet], path: str) -> None:
    try:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        data = [snippet.__dict__ for snippet in snippets]
        with target.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except OSError:
        pass
