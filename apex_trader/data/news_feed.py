from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict, Any
import feedparser
import requests


@dataclass
class NewsItem:
    """Individual financial news item."""
    title: str
    summary: str
    source: str
    published_at: str
    url: str
    sentiment_hint: float  # -1.0 (very negative) to +1.0 (very positive)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "summary": self.summary,
            "source": self.source,
            "published_at": self.published_at,
            "sentiment_hint": self.sentiment_hint,
        }


class NewsFeedCollector:
    """Collects real-time financial news from RSS feeds and Finnhub API."""

    RSS_FEEDS = {
        "crypto": [
            "https://www.coindesk.com/arc/outboundfeeds/rss/",
            "https://cointelegraph.com/rss",
        ],
        "stock": [
            "https://finance.yahoo.com/news/rssindex",
            "https://feeds.content.dowjones.io/public/rss/mw_topstories",
        ],
    }

    BULLISH_KEYWORDS = ["surge", "rally", "breakout", "jump", "record high", "soar", "gain", "bullish", "approval", "adoption", "upgrade", "outperform"]
    BEARISH_KEYWORDS = ["plunge", "crash", "tumble", "dump", "ban", "lawsuit", "investigation", "bearish", "hacked", "inflation", "recession", "downgrade"]

    def __init__(self, finnhub_api_key: Optional[str] = None):
        self.finnhub_api_key = finnhub_api_key

    def fetch_news(self, category: str = "crypto", limit: int = 5) -> List[NewsItem]:
        """Fetches latest news items for a given category (crypto or stock)."""
        items: List[NewsItem] = []

        # 1. Try Finnhub API if key provided
        if self.finnhub_api_key:
            items.extend(self._fetch_finnhub_news(category, limit))

        # 2. Try RSS Feeds
        if len(items) < limit:
            feed_urls = self.RSS_FEEDS.get(category, self.RSS_FEEDS["crypto"])
            for url in feed_urls:
                if len(items) >= limit:
                    break
                items.extend(self._parse_rss_feed(url, limit - len(items)))

        # 3. Fallback dummy curated news if offline/empty
        if not items:
            items = self._get_fallback_news(category)

        return items[:limit]

    def _parse_rss_feed(self, url: str, limit: int) -> List[NewsItem]:
        results = []
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:limit]:
                title = entry.get("title", "")
                summary = entry.get("summary", title)
                link = entry.get("link", "")
                pub_date = entry.get("published", str(datetime.utcnow()))
                
                sentiment = self._calculate_keyword_sentiment(title + " " + summary)
                results.append(NewsItem(
                    title=title,
                    summary=summary[:250],
                    source=feed.feed.get("title", "Financial RSS"),
                    published_at=pub_date,
                    url=link,
                    sentiment_hint=sentiment
                ))
        except Exception:
            pass
        return results

    def _fetch_finnhub_news(self, category: str, limit: int) -> List[NewsItem]:
        results = []
        try:
            cat_param = "crypto" if category == "crypto" else "general"
            url = f"https://finnhub.io/api/v1/news?category={cat_param}&token={self.finnhub_api_key}"
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                for item in data[:limit]:
                    title = item.get("headline", "")
                    summary = item.get("summary", "")
                    sentiment = self._calculate_keyword_sentiment(title + " " + summary)
                    results.append(NewsItem(
                        title=title,
                        summary=summary[:250],
                        source=item.get("source", "Finnhub"),
                        published_at=datetime.fromtimestamp(item.get("datetime", 0)).isoformat(),
                        url=item.get("url", ""),
                        sentiment_hint=sentiment
                    ))
        except Exception:
            pass
        return results

    def _calculate_keyword_sentiment(self, text: str) -> float:
        text_lower = text.lower()
        bull_score = sum(1 for kw in self.BULLISH_KEYWORDS if kw in text_lower)
        bear_score = sum(1 for kw in self.BEARISH_KEYWORDS if kw in text_lower)
        total = bull_score + bear_score
        if total == 0:
            return 0.0
        return (bull_score - bear_score) / total

    def _get_fallback_news(self, category: str) -> List[NewsItem]:
        return [
            NewsItem(
                title=f"Market Update: {category.upper()} Trading Volume Consolidates Near Key Support",
                summary="Traders are watching closely as key liquidity levels hold steady amidst macro developments.",
                source="ApexMarketFeed",
                published_at=datetime.utcnow().isoformat(),
                url="https://example.com/market-news",
                sentiment_hint=0.1
            ),
            NewsItem(
                title="Institutional Inflows Show Resilience in Q3 Trading Sessions",
                summary="Net positive capital flows into major assets indicate solid backing from long-term participants.",
                source="ApexMarketFeed",
                published_at=datetime.utcnow().isoformat(),
                url="https://example.com/institutional-flow",
                sentiment_hint=0.4
            )
        ]
