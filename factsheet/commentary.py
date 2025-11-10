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

DEFAULT_MAX_PER_FEED = 50  # Fetch more articles per feed to ensure 30-day coverage
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
    cutoff_date: str | None = None,
    max_age_days: int = 30,
) -> List[FeedSnippet]:
    """Fetch RSS snippets, optionally filtering by published date.

    Args:
        feeds: RSS feed URLs to fetch from
        max_per_feed: Maximum snippets to fetch per feed
        max_chars: Maximum characters for summary
        cutoff_date: ISO format date string (YYYY-MM-DD). Only include news within max_age_days before this date.
        max_age_days: Maximum age of news items in days (default: 30 days)
    """
    import datetime
    from dateutil import parser as date_parser

    snippets: List[FeedSnippet] = []
    feed_urls = feeds if feeds else DEFAULT_FEEDS

    # Parse cutoff date if provided and calculate date range (max_age_days before as_of to as_of)
    as_of_dt = None
    min_dt = None
    if cutoff_date:
        try:
            as_of_dt = datetime.datetime.fromisoformat(cutoff_date.replace('Z', '+00:00'))
            # Set to end of day for as_of date (include news from that day)
            as_of_dt = as_of_dt.replace(hour=23, minute=59, second=59, microsecond=999999)
            # Calculate minimum date: max_age_days before the as_of date
            min_dt = as_of_dt - datetime.timedelta(days=max_age_days)
            # Set to start of day for clean comparison
            min_dt = min_dt.replace(hour=0, minute=0, second=0, microsecond=0)
            print(f"  Date range: {min_dt.date()} to {as_of_dt.date()}")
        except (ValueError, AttributeError):
            as_of_dt = None
            min_dt = None

    print(f"  Processing {len(feed_urls)} RSS feeds...")

    for idx, feed_url in enumerate(feed_urls, 1):
        feed_prefix = f"  [{idx}/{len(feed_urls)}]"

        try:
            parsed = feedparser.parse(feed_url)
        except Exception as e:
            print(f"{feed_prefix} ✗ Failed to parse: {feed_url[:50]}...")
            continue

        # Check if feed is valid
        if not hasattr(parsed, 'entries') or not parsed.entries:
            print(f"{feed_prefix} ✗ No entries: {feed_url[:50]}...")
            continue

        # Examine up to 200 entries per feed to find articles within date range
        entries = parsed.entries[:200]
        source_name = parsed.feed.get("title", feed_url[:40])

        total_entries = len(entries)
        date_filtered_count = 0
        feed_snippets = []

        for entry in entries:
            # Filter by date if cutoff is specified
            if as_of_dt and min_dt:
                published_str = entry.get("published") or entry.get("updated")
                if published_str:
                    try:
                        published_dt = date_parser.parse(published_str)
                        # Make timezone-naive for comparison
                        if published_dt.tzinfo is not None:
                            published_dt = published_dt.replace(tzinfo=None)
                        # Skip if news is outside the date range [min_dt, as_of_dt]
                        if published_dt < min_dt or published_dt > as_of_dt:
                            date_filtered_count += 1
                            continue
                    except (ValueError, TypeError):
                        # If we can't parse the date, skip the item (safer to exclude than include old news)
                        continue

            title = entry.get("title") or "Untitled"
            summary = entry.get("summary") or entry.get("description") or ""
            summary = _clean_text(summary)
            if len(summary) > max_chars:
                summary = summary[: max_chars - 3].rstrip() + "..."
            published = entry.get("published")
            feed_snippets.append(
                FeedSnippet(source=source_name, title=title, summary=summary, published=published)
            )

            # Stop after collecting max_per_feed valid articles from this feed
            if len(feed_snippets) >= max_per_feed:
                break

        # Report results for this feed
        if feed_snippets:
            print(f"{feed_prefix} ✓ {source_name}: {len(feed_snippets)} articles")
        else:
            if date_filtered_count == total_entries and total_entries > 0:
                print(f"{feed_prefix} ✗ {source_name}: all {total_entries} entries outside date range")
            elif total_entries == 0:
                print(f"{feed_prefix} ✗ {source_name}: feed empty")
            else:
                print(f"{feed_prefix} ✗ {source_name}: 0 valid articles")

        snippets.extend(feed_snippets)

    print(f"\n  Summary: Fetched {len(snippets)} market news articles from {len(feed_urls)} feeds")
    return snippets


def fetch_ticker_news_google(
    holdings: List[Dict[str, Any]],
    max_per_ticker: int = 30,
    max_chars: int = DEFAULT_MAX_SUMMARY_CHARS,
    cutoff_date: str | None = None,
    max_age_days: int = 30,
) -> Dict[str, List[FeedSnippet]]:
    """Fetch news for specific holdings using Google News RSS search.

    Args:
        holdings: List of holdings with 'symbol' or 'yf_symbol' keys
        max_per_ticker: Maximum news items to fetch per ticker (before filtering)
        max_chars: Maximum characters for summary
        cutoff_date: ISO format date string (YYYY-MM-DD). Only include news within max_age_days before this date.
        max_age_days: Maximum age of news items in days (default: 30 days)

    Returns:
        Dictionary mapping ticker symbol to list of news snippets
    """
    import datetime
    from dateutil import parser as date_parser
    from urllib.parse import quote

    news_by_ticker: Dict[str, List[FeedSnippet]] = {}

    # Parse cutoff date for Google News date parameters
    as_of_dt = None
    min_dt = None
    after_param = ""
    before_param = ""

    if cutoff_date:
        try:
            as_of_dt = datetime.datetime.fromisoformat(cutoff_date.replace('Z', '+00:00'))
            as_of_dt = as_of_dt.replace(hour=23, minute=59, second=59, microsecond=999999)
            min_dt = as_of_dt - datetime.timedelta(days=max_age_days)
            min_dt = min_dt.replace(hour=0, minute=0, second=0, microsecond=0)

            # Google News date parameters (YYYY-MM-DD format)
            after_param = f"&after={min_dt.strftime('%Y-%m-%d')}"
            before_param = f"&before={as_of_dt.strftime('%Y-%m-%d')}"
        except (ValueError, AttributeError):
            as_of_dt = None
            min_dt = None

    # Get top 5 holdings
    for holding in holdings[:5]:
        symbol = holding.get('yf_symbol') or holding.get('symbol')
        name = holding.get('name', '')
        if not symbol or symbol in ('CASH', 'USD', 'CASHUSD'):
            continue

        # Build search query for Google News
        base_symbol = symbol.split('.')[0]

        # Create search query: company name OR ticker symbol
        search_terms = []
        if name:
            # Remove common suffixes for cleaner search
            clean_name = name.replace(' Corp.', '').replace(' Inc.', '').replace(' Ltd', '').replace(' Co.', '').strip()
            search_terms.append(f'"{clean_name}"')
        search_terms.append(base_symbol)

        query = ' OR '.join(search_terms)
        encoded_query = quote(query)

        # Google News RSS URL with date filtering
        google_news_url = f"https://news.google.com/rss/search?q={encoded_query}{after_param}{before_param}&hl=en-US&gl=US&ceid=US:en"

        print(f"  Searching Google News for {name} ({base_symbol})")

        try:
            parsed = feedparser.parse(google_news_url)
        except Exception as e:
            print(f"    ✗ Failed to fetch Google News: {e}")
            news_by_ticker[symbol] = []
            continue

        if not hasattr(parsed, 'entries') or not parsed.entries:
            print(f"    ✗ No results from Google News")
            news_by_ticker[symbol] = []
            continue

        entries = parsed.entries[:100]
        ticker_snippets = []

        for entry in entries:
            title = entry.get("title") or "Untitled"
            summary = entry.get("summary") or entry.get("description") or ""

            # Clean up Google News formatting
            summary = _clean_text(summary)
            if len(summary) > max_chars:
                summary = summary[: max_chars - 3].rstrip() + "..."

            published = entry.get("published")

            ticker_snippets.append(
                FeedSnippet(source=f"{name} News (Google)", title=title, summary=summary, published=published)
            )

            if len(ticker_snippets) >= max_per_ticker:
                break

        print(f"    ✓ Found {len(ticker_snippets)} articles from Google News")
        news_by_ticker[symbol] = ticker_snippets

    total_fetched = sum(len(news) for news in news_by_ticker.values())
    successful_tickers = sum(1 for news in news_by_ticker.values() if news)
    print(f"\n  Summary: Fetched {total_fetched} Google News articles for {successful_tickers}/{len(news_by_ticker)} tickers")
    return news_by_ticker


def fetch_ticker_news(
    holdings: List[Dict[str, Any]],
    max_per_ticker: int = 30,
    max_chars: int = DEFAULT_MAX_SUMMARY_CHARS,
    cutoff_date: str | None = None,
    max_age_days: int = 30,
) -> Dict[str, List[FeedSnippet]]:
    """Fetch news for specific holdings/tickers using Yahoo Finance RSS feeds.

    Args:
        holdings: List of holdings with 'symbol' or 'yf_symbol' keys
        max_per_ticker: Maximum news items to fetch per ticker (before filtering)
        max_chars: Maximum characters for summary
        cutoff_date: ISO format date string (YYYY-MM-DD). Only include news within max_age_days before this date.
        max_age_days: Maximum age of news items in days (default: 30 days)

    Returns:
        Dictionary mapping ticker symbol to list of news snippets
    """
    import datetime
    from dateutil import parser as date_parser

    news_by_ticker: Dict[str, List[FeedSnippet]] = {}

    # Parse cutoff date
    as_of_dt = None
    min_dt = None
    if cutoff_date:
        try:
            as_of_dt = datetime.datetime.fromisoformat(cutoff_date.replace('Z', '+00:00'))
            as_of_dt = as_of_dt.replace(hour=23, minute=59, second=59, microsecond=999999)
            min_dt = as_of_dt - datetime.timedelta(days=max_age_days)
            min_dt = min_dt.replace(hour=0, minute=0, second=0, microsecond=0)
        except (ValueError, AttributeError):
            as_of_dt = None
            min_dt = None

    # Get top 5 holdings only (most important)
    for holding in holdings[:5]:
        symbol = holding.get('yf_symbol') or holding.get('symbol')
        name = holding.get('name', '')
        if not symbol or symbol in ('CASH', 'USD', 'CASHUSD'):
            continue

        # Yahoo Finance RSS feed for specific ticker
        feed_url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US"

        try:
            parsed = feedparser.parse(feed_url)
        except Exception as e:
            print(f"  - Failed to parse feed for {symbol}: {e}")
            continue

        # Check if feed is valid
        if not hasattr(parsed, 'entries'):
            print(f"  - Feed for {symbol} returned no data structure")
            continue

        entries = parsed.entries
        print(f"  - Feed returned {len(entries)} total entries for {symbol}")

        entries = entries[:100]  # Check more entries to find valid ones
        ticker_snippets = []

        articles_checked = 0
        articles_date_filtered = 0

        for entry in entries:
            articles_checked += 1

            # Filter by date if cutoff is specified - STRICT filtering
            if as_of_dt and min_dt:
                published_str = entry.get("published") or entry.get("updated")
                if published_str:
                    try:
                        published_dt = date_parser.parse(published_str)
                        if published_dt.tzinfo is not None:
                            published_dt = published_dt.replace(tzinfo=None)
                        # Only accept news within the specified date range
                        if published_dt < min_dt or published_dt > as_of_dt:
                            articles_date_filtered += 1
                            continue
                    except (ValueError, TypeError):
                        pass  # If we can't parse date, allow the article

            title = entry.get("title") or "Untitled"
            summary = entry.get("summary") or entry.get("description") or ""

            # Since we're fetching from Yahoo Finance RSS for this specific ticker,
            # we can trust that most articles are relevant. Only filter out obviously unrelated ones.
            # Just do a basic sanity check - if the article has ANY content, include it
            if not title or title == "Untitled":
                continue

            summary = _clean_text(summary)
            if len(summary) > max_chars:
                summary = summary[: max_chars - 3].rstrip() + "..."
            published = entry.get("published")

            ticker_snippets.append(
                FeedSnippet(source=f"{name} News", title=title, summary=summary, published=published)
            )

            if len(ticker_snippets) >= max_per_ticker:
                break

        if ticker_snippets:
            print(f"✓ Fetched {len(ticker_snippets)} articles for {name} ({symbol})")
            news_by_ticker[symbol] = ticker_snippets
        else:
            print(f"✗ No articles found for {name} ({symbol})")
            print(f"  - Checked {articles_checked} entries from feed")
            print(f"  - {articles_date_filtered} filtered out by date range")
            if articles_checked == 0:
                print(f"  - Feed returned no entries - RSS feed might be empty or unavailable")
            news_by_ticker[symbol] = []

    total_fetched = sum(len(news) for news in news_by_ticker.values())
    successful_tickers = sum(1 for news in news_by_ticker.values() if news)
    print(f"\n  Summary: Fetched {total_fetched} Yahoo Finance articles for {successful_tickers}/{len(news_by_ticker)} tickers")
    return news_by_ticker


def filter_news_per_ticker_with_ai(
    ticker_symbol: str,
    ticker_name: str,
    snippets: List[FeedSnippet],
    target_count: int,
    api_key: str,
    model: str = "gpt-4o-mini",
) -> List[FeedSnippet]:
    """Use AI to select the most relevant news articles for a specific ticker.

    Args:
        ticker_symbol: Stock ticker symbol
        ticker_name: Company name
        snippets: List of news snippets for this ticker
        target_count: Number of articles to select
        api_key: OpenAI API key
        model: OpenAI model to use

    Returns:
        Filtered list of most relevant snippets
    """
    if not snippets or len(snippets) <= target_count:
        return list(snippets)

    if not api_key:
        return list(snippets[:target_count])

    # Create numbered list of news items
    news_items = []
    for idx, snippet in enumerate(snippets):
        news_items.append(f"{idx}. {snippet.title} - {snippet.summary[:150]}")

    prompt = f"""You are filtering news for {ticker_name} ({ticker_symbol}).

News Items (numbered 0-{len(snippets)-1}):
{chr(10).join(news_items)}

Select the {target_count} MOST RELEVANT and IMPACTFUL news articles about {ticker_name} that would help explain:
- Company performance, earnings, or financial results
- Major business developments or strategic changes
- Industry trends affecting this company
- Analyst opinions or ratings

Exclude generic market news unless it specifically impacts this company.

Respond with ONLY a JSON array of exactly {target_count} numbers, e.g., [0, 2, 5, 8, 11]"""

    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=100,
        )
        content = response.choices[0].message.content.strip()

        # Parse JSON array from response
        import json
        import re
        match = re.search(r'\[[\d,\s]+\]', content)
        if match:
            selected_indices = json.loads(match.group(0))
            selected_indices = [i for i in selected_indices if 0 <= i < len(snippets)]
            return [snippets[i] for i in selected_indices[:target_count]]
    except Exception as e:
        print(f"  Warning: AI filtering failed for {ticker_symbol}: {e}")

    # Fallback: return first target_count items
    return list(snippets[:target_count])


def extract_ticker_news_from_market_feeds(
    holdings: List[Dict[str, Any]],
    market_snippets: Sequence[FeedSnippet],
    max_per_ticker: int = 30,
) -> Dict[str, List[FeedSnippet]]:
    """Extract news about specific holdings from general market news feeds.

    This is a fallback method when direct ticker RSS feeds don't work well.

    Returns:
        Dictionary mapping ticker symbol to list of news snippets
    """
    news_by_ticker: Dict[str, List[FeedSnippet]] = {}
    used_titles: set = set()  # Track to avoid duplicates

    # Get top 5 holdings
    for holding in holdings[:5]:
        symbol = holding.get('yf_symbol') or holding.get('symbol')
        name = holding.get('name', '')
        if not symbol or symbol in ('CASH', 'USD', 'CASHUSD'):
            continue

        # Build search keywords - be aggressive to find all mentions
        search_terms = set()

        # Add ticker symbols in various formats
        base_symbol = symbol.split('.')[0]
        search_terms.add(symbol.lower())
        search_terms.add(base_symbol.lower())
        search_terms.add(f"${base_symbol.lower()}")  # Stock mentions like $NVDA
        search_terms.add(f"({base_symbol.lower()})")  # Mentions like (NVDA)

        # Add company name keywords with more aggressive matching
        if name:
            common_words = {'inc', 'corp', 'ltd', 'co', 'company', 'limited', 'plc', 'llc', 'group', 'adr', 'the', '-', 'class'}

            # Full company name
            search_terms.add(name.lower())

            # Individual words from company name
            for word in name.lower().replace('.', ' ').replace(',', ' ').replace('-', ' ').split():
                clean_word = word.strip('.').strip()
                # Use 3+ char keywords for better coverage
                if len(clean_word) >= 3 and clean_word not in common_words:
                    search_terms.add(clean_word)

            # Special handling for well-known companies
            # Add common abbreviations and alternative names
            if 'nvidia' in name.lower():
                search_terms.add('nvidia')
                search_terms.add('nvda')
            if 'jpmorgan' in name.lower() or 'chase' in name.lower():
                search_terms.add('jpmorgan')
                search_terms.add('chase')
                search_terms.add('jpm')
            if 'kimberly' in name.lower():
                search_terms.add('kimberly')
                search_terms.add('kimberly-clark')
                search_terms.add('kmb')
            if 'visa' in name.lower():
                search_terms.add('visa')

        print(f"  Searching for {name} using keywords: {', '.join(sorted(search_terms))}")

        # Search through market news
        holding_news = []
        matches_found = 0
        for snippet in market_snippets:
            if snippet.title in used_titles:
                continue

            combined_text = (snippet.title + " " + snippet.summary).lower()

            # Check if any search term appears
            matched_terms = [term for term in search_terms if term in combined_text]
            if matched_terms:
                matches_found += 1
                # Create a new snippet with ticker-specific source label
                ticker_snippet = FeedSnippet(
                    source=f"{name} News",
                    title=snippet.title,
                    summary=snippet.summary,
                    published=snippet.published,
                )
                holding_news.append(ticker_snippet)
                used_titles.add(snippet.title)

                if len(holding_news) >= max_per_ticker:
                    break

        if holding_news:
            print(f"  ✓ Extracted {len(holding_news)} articles for {name} from {len(market_snippets)} market articles")
        else:
            print(f"  ✗ No articles found for {name} in {len(market_snippets)} market articles")

        news_by_ticker[symbol] = holding_news

    total_extracted = sum(len(news) for news in news_by_ticker.values())
    successful_tickers = sum(1 for news in news_by_ticker.values() if news)
    print(f"\n  Summary: Extracted {total_extracted} articles for {successful_tickers}/{len(news_by_ticker)} tickers from market feeds")
    return news_by_ticker


def filter_snippets_with_ai(
    snippets: Sequence[FeedSnippet],
    context: Dict[str, Any],
    config: Dict[str, Any],
) -> List[FeedSnippet]:
    """Use OpenAI to filter news snippets for relevance based on portfolio and market insights.

    Args:
        snippets: List of news snippets to filter
        context: Portfolio context with holdings, sectors, regions
        config: Configuration with OpenAI settings

    Returns:
        List of filtered snippets selected by AI as most relevant
    """
    if not snippets:
        return []

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return list(snippets)

    # Build context description
    holdings = context.get("holdings", [])
    top_holdings = ", ".join(f"{h['name']} ({h.get('weight_pct', 0):.1f}%)" for h in holdings[:5])
    sectors = context.get("sectors", [])
    sector_desc = ", ".join(f"{name}: {value:.1%}" for name, value in sectors[:5])

    # Create numbered list of news items
    news_items = []
    for idx, snippet in enumerate(snippets):
        news_items.append(f"{idx}. [{snippet.source}] {snippet.title} - {snippet.summary[:200]}")

    max_headlines = config.get("max_headlines", 4)

    prompt = f"""You are an investment analyst filtering news for a portfolio commentary.

Portfolio Context:
- Top Holdings: {top_holdings}
- Sector Allocation: {sector_desc}

News Items (numbered 0-{len(snippets)-1}):
{chr(10).join(news_items)}

Task: Select EXACTLY {max_headlines} news items that provide the best mix of:
1. SPECIFIC news about the portfolio's top holdings (at least 50% should be ticker-specific)
2. GENERAL market/economic news that provides context (at least 30% should be market-wide)

Selection criteria:
- Prioritize news that explains WHY holdings performed well/poorly
- Include market context that affects this portfolio's sectors
- Be STRICT - exclude generic noise that doesn't matter
- Ensure you select from BOTH ticker-specific AND general market news

Respond with ONLY a JSON array of exactly {max_headlines} numbers, e.g., [0, 3, 7, 12]"""

    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=config.get("model", "gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=100,
        )
        content = response.choices[0].message.content.strip()

        # Parse JSON array from response
        import json
        import re
        # Extract JSON array pattern
        match = re.search(r'\[[\d,\s]+\]', content)
        if match:
            selected_indices = json.loads(match.group(0))
            # Filter to valid indices
            selected_indices = [i for i in selected_indices if 0 <= i < len(snippets)]
            # Return selected snippets
            return [snippets[i] for i in selected_indices[:max_headlines]]
    except Exception:
        # If AI filtering fails, fall back to returning first max_headlines
        pass

    return list(snippets[:max_headlines])


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
        "You are crafting a concise manager's commentary for an investment factsheet.",
        "Write EXACTLY 150-160 words in a professional, balanced tone with two to three short paragraphs.",
        "",
        "STRUCTURE:",
        "1. First paragraph (60-70 words): Discuss the main drivers of portfolio performance. Focus on WHY the portfolio performed as it did - identify which holdings or sectors were primary contributors or detractors and explain the reasons based on the news provided.",
        "2. Second paragraph (50-60 words): Broader market context and outlook. Discuss relevant market trends, economic factors, or sector dynamics that affected or may affect the portfolio.",
        "3. Optional brief third paragraph (30-40 words): Forward-looking considerations or positioning if relevant.",
        "",
        "CRITICAL REQUIREMENTS:",
        "- DO NOT repeat exact performance numbers (YTD, since inception, CAGR) - these are shown elsewhere on the factsheet",
        "- Focus on the 'WHY' not the 'WHAT' - explain drivers, not just results",
        "- Use the news headlines to explain performance drivers",
        "- Write in plain paragraphs (no bullet points)",
        "- MUST complete all sentences - do NOT stop mid-sentence",
        "- Stay within 150-160 words to ensure completion",
        "",
        "CONTEXT:",
        f"Factsheet Date: {as_of}",
        f"Portfolio YTD: {portfolio.get('ytd')} (for context only - don't repeat in commentary)",
        f"Benchmark YTD: {benchmark.get('ytd')} (for context only - don't repeat in commentary)",
    ]

    if sectors:
        sector_lines = ", ".join(f"{name}: {value:.1%}" for name, value in sectors[:5])
        lines.append(f"Top Sector Allocations: {sector_lines}")
    if regions:
        region_lines = ", ".join(f"{name}: {value:.1%}" for name, value in regions[:3])
        lines.append(f"Regional Allocations: {region_lines}")
    if holdings:
        holdings_text = ", ".join(f"{item['name']} ({item['weight_pct']:.1f}%)" for item in holdings[:5])
        lines.append(f"Top 5 Holdings: {holdings_text}")

    if snippets:
        lines.append("")
        lines.append("RECENT NEWS (use these to explain performance drivers):")
        for snippet in snippets:
            lines.append(snippet.to_prompt_fragment())

    lines.append("")
    lines.append("Write the commentary now (150-160 words, complete sentences only):")
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

    # Extract as_of date from context to filter news
    as_of_date = context.get("as_of")
    # Get max age from config (default: 30 days)
    max_age_days = _coerce_int(config.get("max_news_age_days"), 30) or 30

    # Fetch general market news FIRST - this is important for market context
    print(f"\nFetching general market news...")
    try:
        market_snippets = fetch_rss_snippets(
            rss_feeds,
            max_per_feed=max_per_feed,
            max_chars=max_summary_chars,
            cutoff_date=as_of_date,
            max_age_days=max_age_days,
        )
    except Exception as e:
        print(f"Warning: Failed to fetch market news: {e}")
        market_snippets = []

    # Get news_per_ticker config
    news_per_ticker = _coerce_int(config.get("news_per_ticker"), 5) or 5
    holdings = context.get("holdings", [])

    # 1. Fetch from Google News (best date filtering and search)
    print(f"\nFetching ticker news from Google News RSS...")
    try:
        google_news_dict = fetch_ticker_news_google(
            holdings,
            max_per_ticker=30,  # Fetch up to 30, will filter to news_per_ticker later
            max_chars=max_summary_chars,
            cutoff_date=as_of_date,
            max_age_days=max_age_days,
        )
    except Exception as e:
        print(f"Warning: Failed to fetch Google News: {e}")
        google_news_dict = {}

    # 2. Fetch from Yahoo Finance RSS (ticker-specific feeds)
    print(f"\nFetching ticker-specific news from Yahoo Finance RSS...")
    try:
        ticker_news_dict = fetch_ticker_news(
            holdings,
            max_per_ticker=30,  # Fetch up to 30, will filter to news_per_ticker later
            max_chars=max_summary_chars,
            cutoff_date=as_of_date,
            max_age_days=max_age_days,
        )
    except Exception as e:
        print(f"Warning: Failed to fetch ticker news: {e}")
        ticker_news_dict = {}

    # 3. FALLBACK: Extract ticker-relevant news from market feeds
    # This helps when direct ticker feeds don't work well
    print(f"\nExtracting ticker-relevant news from market feeds (fallback)...")
    try:
        extracted_news_dict = extract_ticker_news_from_market_feeds(
            holdings,
            market_snippets,
            max_per_ticker=30,  # Fetch up to 30, will filter later
        )
    except Exception as e:
        print(f"Warning: Failed to extract ticker news from market feeds: {e}")
        extracted_news_dict = {}

    # Combine news from all sources per ticker
    print(f"\nApplying AI filtering per ticker (selecting top {news_per_ticker} articles per ticker)...")
    filtered_ticker_news = []
    model = config.get("model", "gpt-4o-mini")

    for holding in holdings[:5]:
        symbol = holding.get('yf_symbol') or holding.get('symbol')
        name = holding.get('name', '')
        if not symbol or symbol in ('CASH', 'USD', 'CASHUSD'):
            continue

        # Combine news from all three sources for this ticker
        ticker_news = (
            google_news_dict.get(symbol, []) +
            ticker_news_dict.get(symbol, []) +
            extracted_news_dict.get(symbol, [])
        )

        if not ticker_news:
            print(f"  • {name} ({symbol}): No news found")
            continue

        # Apply AI filtering to select best news_per_ticker articles
        filtered = filter_news_per_ticker_with_ai(
            ticker_symbol=symbol,
            ticker_name=name,
            snippets=ticker_news,
            target_count=news_per_ticker,
            api_key=api_key,
            model=model,
        )

        print(f"  • {name} ({symbol}): {len(ticker_news)} → {len(filtered)} articles")
        filtered_ticker_news.extend(filtered)

    # Combine ticker news with market news
    all_snippets = filtered_ticker_news + market_snippets
    print(f"\n  Summary: {len(filtered_ticker_news)} ticker-specific + {len(market_snippets)} market-general = {len(all_snippets)} total articles for final filtering")

    news_cache_path = config.get("news_cache")
    if news_cache_path:
        save_news_cache(all_snippets, news_cache_path)

    # Apply AI-based filtering to select most relevant from combined pool
    print(f"\nApplying final AI filtering to select best articles for commentary...")
    filtered_snippets: List[FeedSnippet]
    snippet_debug: List[Dict[str, Any]]

    if len(all_snippets) > 0:
        try:
            # AI filtering to select most relevant articles
            ai_filtered = filter_snippets_with_ai(all_snippets, context, config)
            # Don't apply keyword filtering after AI - let AI make the final call
            filtered_snippets = ai_filtered
            snippet_debug = [{"source": s.source, "title": s.title, "published": s.published} for s in ai_filtered]
            print(f"  ✓ Selected {len(filtered_snippets)} articles for commentary generation")
        except Exception as e:
            print(f"Warning: AI filtering failed: {e}")
            # Fallback: take first few from each category
            max_headlines = config.get("max_headlines", 4)
            # Take mix: 60% ticker news, 40% market news
            ticker_count = min(len(ticker_snippets), max(1, int(max_headlines * 0.6)))
            market_count = max_headlines - ticker_count
            filtered_snippets = ticker_snippets[:ticker_count] + market_snippets[:market_count]
            snippet_debug = []
    else:
        filtered_snippets = []
        snippet_debug = []

    prompt = build_prompt(context, filtered_snippets)
    client = OpenAI(api_key=api_key)
    model = config.get("model", "gpt-4o-mini")
    # Set higher max_tokens (300) to ensure completion without mid-sentence cutoff
    # The prompt constrains output to 150-160 words, but this gives buffer for completion
    max_tokens = config.get("max_tokens", 300)
    temperature = config.get("temperature", 0.6)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        text = response.choices[0].message.content.strip()
        finish_reason = response.choices[0].finish_reason

        # Warn if the response was truncated due to max_tokens
        if finish_reason == "length":
            # Log a warning but still return the text
            # In production, you might want to retry with higher max_tokens
            pass
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
        "fetched_snippets": [snippet.__dict__ for snippet in all_snippets],
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
