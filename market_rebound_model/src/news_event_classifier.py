"""Deterministic event features for causal rebound-model news V3."""
from __future__ import annotations

import re

import pandas as pd

# Keep these patterns deliberately focused on article-language terms.  Generic
# GKG taxonomy labels such as ECON_STOCKMARKET are not sufficient evidence of
# an event family and previously caused too many rows to collapse into noisy
# classifications.
EVENT_PATTERNS = {
    "earnings": r"\b(earnings|revenue|profit|eps|quarterly|results|financial-results|q[1-4])\b",
    "guidance": r"\b(guidance|outlook|forecast|raised-guidance|cut-guidance|cuts-guidance)\b",
    "analyst": r"\b(upgrade|downgrade|price-target|analyst|rating|target-price)\b",
    "regulatory": r"\b(regulator|regulatory|antitrust|lawsuit|investigation|fine|sec|probe|settlement|court|legal)\b",
    "ma": r"\b(acquire|acquisition|acquires|merger|merges|takeover|deal|buyout)\b",
    "product": r"\b(product|launch|recall|delivery|deliveries|production|manufacturing|model|vehicle|truck|car)\b",
    "macro": r"\b(fed|inflation|interest-rate|rates|tariff|recession|gdp|jobs-report|unemployment|monetary-policy)\b",
}
EVENT_TYPES = tuple(EVENT_PATTERNS)

# High-confidence GKG theme tokens used only when the article-language fields
# do not identify an event. We intentionally avoid generic stock-market tags.
THEME_EVENT_TOKENS = {
    "earnings": ("ECON_EARNINGS",),
    "guidance": (),
    "analyst": (),
    "regulatory": ("EPU_POLICY_REGULATORY",),
    "ma": (),
    "product": (),
    "macro": (
        "EPU_CATS_MONETARY_POLICY",
        "WB_444_MONETARY_POLICY",
        "WB_439_MACROECONOMIC_AND_STRUCTURAL_POLICIES",
        "EPU_POLICY_MONETARY_POLICY",
        "WB_1104_MACROECONOMIC_VULNERABILITY_AND_DEBT",
    ),
}

NEGATIVE_WORDS = re.compile(
    r"\b(miss|misses|weak|cut|cuts|lower|warning|decline|falls?|drop|plunge|loss|"
    r"lawsuit|investigation|fine|downgrade|slump|slowdown)\b",
    re.I,
)
POSITIVE_WORDS = re.compile(
    r"\b(beat|beats|strong|raise|raises|higher|growth|profit|upgrade|buy|approval|"
    r"record|rebound)\b",
    re.I,
)


def _normalize_article_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _looks_like_gkg_theme_string(text: str) -> bool:
    """Identify the semicolon-delimited uppercase taxonomy strings used by GKG."""
    if not text or ";" not in text:
        return False
    parts = [part.strip() for part in text.split(";")]
    return bool(parts) and all(
        part and re.fullmatch(r"[A-Z0-9]+(?:_[A-Z0-9]+)+", part) for part in parts
    )


def _event_matches(text: str) -> list[str]:
    low = text.lower()
    return [name for name, pattern in EVENT_PATTERNS.items() if re.search(pattern, low)]


def classify_headline(text: str) -> tuple[str, float, float]:
    """Backward-compatible classifier for a single text string."""
    text = str(text or "")
    matches = _event_matches(text)
    event = matches[0] if matches else "other"
    neg = len(NEGATIVE_WORDS.findall(text))
    pos = len(POSITIVE_WORDS.findall(text))
    polarity = (pos - neg) / max(pos + neg, 1)
    event_intensity = min(1.0, (len(matches) + neg + pos) / 4.0)
    return event, polarity, event_intensity


def classify_article(headline: str, summary: str, url: str) -> tuple[str, float, float, str]:
    """Classify an article using high-confidence text before GKG themes.

    Priority is headline > URL slug > natural-language summary > specific GKG
    theme token. Broad taxonomy labels such as ECON_STOCKMARKET are ignored as
    direct event evidence.
    """
    headline = _normalize_article_text(headline)
    summary = _normalize_article_text(summary)
    url_text = re.sub(r"[-_/+?.=&%]+", " ", _normalize_article_text(url))

    source_texts: list[tuple[int, str, str]] = [
        (6, "headline", headline),
        (4, "url", url_text),
    ]
    if summary and not _looks_like_gkg_theme_string(summary):
        source_texts.append((2, "summary", summary))

    candidates: list[tuple[int, int, str, str]] = []
    for source_weight, source_name, text in source_texts:
        if not text:
            continue
        for event_index, (event, pattern) in enumerate(EVENT_PATTERNS.items()):
            if re.search(pattern, text.lower()):
                # Lower event_index wins ties; this keeps the existing event
                # order deterministic when one text mentions two event families.
                candidates.append((source_weight, -event_index, event, source_name))

    if candidates:
        candidates.sort(reverse=True)
        _, _, event, source = candidates[0]
        classification_source = source
    else:
        summary_upper = summary.upper()
        event = "other"
        classification_source = "other"
        for candidate in EVENT_TYPES:
            tokens = THEME_EVENT_TOKENS.get(candidate, ())
            if any(token in summary_upper for token in tokens):
                event = candidate
                classification_source = "gkg_theme"
                break

    # Sentiment remains text-based, using article language/slug first and GKG
    # themes only as a fallback when no article language is available.
    sentiment_text = " ".join(x for x in (headline, url_text, summary) if x)
    neg = len(NEGATIVE_WORDS.findall(sentiment_text))
    pos = len(POSITIVE_WORDS.findall(sentiment_text))
    polarity = (pos - neg) / max(pos + neg, 1)
    direct_matches = len(_event_matches(" ".join(x for x in (headline, url_text) if x)))
    if classification_source == "summary" and summary:
        direct_matches = max(direct_matches, len(_event_matches(summary)))
    theme_match = 1 if classification_source == "gkg_theme" else 0
    event_intensity = min(1.0, (direct_matches + theme_match + neg + pos) / 4.0)
    return event, polarity, event_intensity, classification_source


def add_event_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ("headline", "summary", "url"):
        if col not in out.columns:
            out[col] = ""

    classified = out.apply(
        lambda row: classify_article(row["headline"], row["summary"], row["url"]),
        axis=1,
        result_type="expand",
    )
    classified.columns = ["event_type", "event_polarity", "event_intensity", "event_source"]

    headline = out["headline"].fillna("").astype(str).str.strip()
    summary = out["summary"].fillna("").astype(str).str.strip()
    url = out["url"].fillna("").astype(str).str.strip()
    text = headline.where(headline.ne(""), "")
    text = text + " " + summary + " " + url.str.replace(r"[-_/+?.=&%]+", " ", regex=True)
    text = text.str.replace(r"\s+", " ", regex=True).str.strip()

    out["classification_text"] = text
    out["event_type"] = classified["event_type"]
    out["event_polarity"] = pd.to_numeric(classified["event_polarity"], errors="coerce").fillna(0.0)
    out["event_intensity"] = pd.to_numeric(classified["event_intensity"], errors="coerce").fillna(0.0)
    out["event_source"] = classified["event_source"]
    out["is_negative_event"] = (out["event_polarity"] < 0).astype(int)
    out["is_material_event"] = (out["event_intensity"] >= 0.5).astype(int)
    return out


def build_daily_event_features(df: pd.DataFrame) -> pd.DataFrame:
    x = add_event_features(df)
    x["Date"] = pd.to_datetime(x["published_at"], utc=True).dt.normalize()
    keys = ["Date", "symbol"]
    daily = (
        x.groupby(keys, as_index=False)
        .agg(
            news_count=("classification_text", "size"),
            negative_news_share=("is_negative_event", "mean"),
            material_event_share=("is_material_event", "mean"),
            event_polarity=("event_polarity", "mean"),
            event_intensity=("event_intensity", "mean"),
            unique_event_types=("event_type", "nunique"),
        )
    )
    event_counts = pd.crosstab([x["Date"], x["symbol"]], x["event_type"]).reset_index()
    event_counts = event_counts.rename(
        columns={c: f"event_{c}_share" for c in event_counts.columns if c not in keys}
    )
    daily = daily.merge(event_counts, on=keys, how="left")
    for event in EVENT_TYPES + ("other",):
        col = f"event_{event}_share"
        if col not in daily.columns:
            daily[col] = 0.0
        daily[col] = pd.to_numeric(daily[col], errors="coerce").fillna(0.0)
        daily[col] = daily[col] / daily["news_count"].clip(lower=1)
    return daily
