import unittest
from datetime import datetime, timezone
import pandas as pd

from news_provider import NewsQuery, validate_articles
from finnhub_news import FinnhubNewsProvider

class NewsPipelineTests(unittest.TestCase):
    def test_timezone_safe_cutoff(self):
        q = NewsQuery(
            "NVDA",
            datetime(2026, 8, 1, tzinfo=timezone.utc),
            datetime(2026, 8, 2, tzinfo=timezone.utc),
        )
        df = pd.DataFrame([
            {"published_at":"2026-08-01T12:00:00Z","symbol":"NVDA","headline":"inside","source":"x","url":"u1","summary":"","category":"company","sentiment":0.1,"intensity":0.1,"relevance":0.8,"novelty":None},
            {"published_at":"2026-08-02T00:00:01Z","symbol":"NVDA","headline":"outside","source":"x","url":"u2","summary":"","category":"company","sentiment":0.1,"intensity":0.1,"relevance":0.8,"novelty":None},
        ])
        out = validate_articles(df, q)
        self.assertEqual(len(out), 1)
        self.assertEqual(out.iloc[0]["headline"], "inside")

    def test_finnhub_requires_key(self):
        with self.assertRaises(RuntimeError):
            FinnhubNewsProvider(api_key=None)

if __name__ == "__main__":
    unittest.main()
