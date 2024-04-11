from datetime import datetime, timedelta

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.agents import tool
from openbb import obb
import quantstats as qs
import pandas as pd

from app.features.technical import add_technicals
from app.tools.utils import wrap_dataframe


class StockStatsInput(BaseModel):
    symbol: str = Field(..., description="Stock symbol to fetch data for")


@tool(args_schema=StockStatsInput)
def get_stock_price_history(symbol: str) -> str:
    """Fetch a Stock's Price History by Symbol."""

    try:
        start_date = (datetime.now() - timedelta(days=365 * 2)).strftime("%Y-%m-%d")
        df = obb.equity.price.historical(
            symbol, start_date=start_date, provider="yfinance"
        ).to_df()
        df.index = pd.to_datetime(df.index)

        if df.empty:
            return "\n<observation>\nNo data found for the given symbol</observation>\n"

        df = add_technicals(df)
        df = df[-30:][::-1]

        return wrap_dataframe(df)
    except Exception as e:
        return f"\n<observation>\nError: {e}</observation>\n"


@tool(args_schema=StockStatsInput)
def get_stock_quantstats(symbol: str) -> str:
    """Fetch a Stock's Portfolio Analytics For Quants by Symbol."""

    try:
        start_date = (datetime.now() - timedelta(days=365 * 2)).strftime("%Y-%m-%d")
        df = obb.equity.price.historical(
            symbol, start_date=start_date, provider="yfinance"
        ).to_df()
        df.index = pd.to_datetime(df.index)

        if df.empty:
            return f"\n<observation>\nNo data found for the given symbol {symbol}</observation>\n"

        stock_ret = qs.utils.download_returns(symbol, period=df.index)
        bench_ret = qs.utils.download_returns("^GSPC", period=df.index)
        stats = qs.reports.metrics(
            stock_ret, mode="full", benchmark=bench_ret, display=False
        )

        return f"\n<observation>\n\n{stats}\n\n</observation>\n"
    except Exception as e:
        return f"\n<observation>\nError: {e}\n\n</observation>\n"


@tool
def get_gainers() -> str:
    """Fetch Top Price Gainers in the Stock Market."""

    try:
        gainers = obb.equity.discovery.gainers(sort="desc").to_df()

        if gainers.empty:
            return "\n<observation>\nNo gainers found</observation>\n"

        return wrap_dataframe(gainers)
    except Exception as e:
        return f"\n<observation>\nError: {e}</observation>\n"


@tool
def get_losers() -> str:
    """Fetch Stock Market's Top Losers."""

    try:
        losers = obb.equity.discovery.losers(sort="desc").to_df()

        if losers.empty:
            return "\n<observation>\nNo losers found</observation>\n"

        return wrap_dataframe(losers)
    except Exception as e:
        return f"\n<observation>\nError: {e}</observation>\n"


@tool(args_schema=StockStatsInput)
def get_stock_ratios(symbol: str) -> str:
    """Fetch an Extensive Set of Financial and Accounting Ratios for a Given Company Over Time."""

    try:
        trades = obb.equity.fundamental.ratios(symbol=symbol).to_df()

        if trades.empty:
            return f"\n<observation>\nNo data found for the given symbol {symbol}</observation>\n"

        return wrap_dataframe(trades)
    except Exception as e:
        return f"\n<observation>\nError: {e}</observation>\n"


@tool(args_schema=StockStatsInput)
def get_key_metrics(symbol: str) -> str:
    """Fetch Fundamental Metrics by Symbol."""

    try:
        metrics = obb.equity.fundamental.metrics(symbol=symbol, with_ttm=True).to_df()[
            ::-1
        ]

        if metrics.empty:
            return f"\n<observation>\nNo data found for the given symbol {symbol}</observation>\n"

        return wrap_dataframe(metrics)
    except Exception as e:
        return f"\n<observation>\nError: {e}</observation>\n"


@tool(args_schema=StockStatsInput)
def get_stock_sector_info(symbol: str) -> str:
    """Fetch a Company's General Information By Symbol. This includes company name, industry, and sector data."""

    try:
        profile = obb.equity.profile(symbol=symbol).to_df()

        if profile.empty:
            return f"\n<observation>\nNo data found for the given symbol {symbol}</observation>\n"

        return wrap_dataframe(profile)
    except Exception as e:
        return f"\n<observation>\nError: {e}</observation>\n"


@tool(args_schema=StockStatsInput)
def get_valuation_multiples(symbol: str) -> str:
    """Fetch a Company's Valuation Multiples by Symbol."""

    try:
        df = obb.equity.fundamental.multiples(symbol=symbol).to_df()

        if df.empty:
            return f"\n<observation>\nNo data found for the given symbol {symbol}</observation>\n"

        return wrap_dataframe(df)
    except Exception as e:
        return f"\n<observation>\nError: {e}</observation>\n"
