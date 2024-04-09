from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.agents import tool
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from openbb import obb

from app.tools.utils import wrap_dataframe


class StockStatsInput(BaseModel):
    symbol: str = Field(..., description="Stock symbol to fetch data for")


def analyze_sentiment(text):
    if text is None:
        return 0.0

    try:
        analyzer = SentimentIntensityAnalyzer()
        sentiment = analyzer.polarity_scores(text)
        if "compound" in sentiment:
            return sentiment["compound"]
        else:
            return 0.0
    except Exception:
        return 0.0


@tool
def get_news_sentiment(symbol: str) -> str:
    """Get News Sentiment for a Stock."""

    try:
        df = obb.news.company(symbol=symbol, provider="tiingo", limit=10).to_df()

        if df.empty:
            return "\n<observation>\nNo data found for the given symbol</observation>\n"

        if "text" in df.columns:
            df["sentiment_score"] = df["text"].apply(analyze_sentiment)
        else:
            df["sentiment_score"] = df["title"].apply(analyze_sentiment)

        return wrap_dataframe(df)
    except Exception as e:
        return f"\n<observation>\nError: {e}</observation>\n"
