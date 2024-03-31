from datetime import datetime, timedelta

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.agents import tool
from openbb import obb
import quantstats as qs
import pandas as pd

from app.features.technical import add_technicals


class StockStatsInput(BaseModel):
    symbol: str = Field(..., description="Stock symbol to fetch data for")


@tool(args_schema=StockStatsInput)
def get_stock_stats(symbol: str) -> str:
    """Get Stock History and Statistics by Symbol."""

    try:
        start_date = (datetime.now() - timedelta(days=365 * 2)).strftime("%Y-%m-%d")
        df = obb.equity.price.historical(
            symbol, start_date=start_date, provider="yfinance"
        ).to_df()
        df.index = pd.to_datetime(df.index)

        if df.empty:
            return f"No data found for the given symbol {symbol}"

        stock_ret = qs.utils.download_returns(symbol, period=df.index)
        bench_ret = qs.utils.download_returns("^GSPC", period=df.index)
        stats = qs.reports.metrics(
            stock_ret, mode="full", benchmark=bench_ret, display=False
        )

        df = add_technicals(df)
        df = df[-100:][::-1]

        return f"Stats for {symbol}:\n{stats}\n\n{df.to_string(index=False)}"
    except Exception as e:
        return f"Error: {e}"


@tool
def get_gainers() -> str:
    """Get Top Gainers."""

    try:
        gainers = obb.equity.discovery.gainers(sort="desc").to_df()

        if gainers.empty:
            return "No gainers found"

        return f"Top Gainers:\n{gainers.to_string(index=False)}"
    except Exception as e:
        return f"Error: {e}"


@tool
def get_losers() -> str:
    """Get Top Losers."""

    try:
        losers = obb.equity.discovery.losers(sort="desc").to_df()

        if losers.empty:
            return "No losers found"

        return f"Top Losers:\n{losers.to_string(index=False)}"
    except Exception as e:
        return f"Error: {e}"
