from datetime import datetime, timedelta

from langchain.agents import tool
from openbb import obb
from pandas_datareader import data as pdr

from app.features.technical import add_technicals
from app.features.screener import fetch_custom_universe
from app.tools.utils import wrap_dataframe
from app.tools.types import StockStatsInput

import quantstats as qs
import pandas as pd
import yfinance as yf

yf.pdr_override()


def fetch_and_convert_ohlc(symbol: str, start_date: str) -> pd.DataFrame:
    """
    Fetch stock data using pandas_datareader and convert OHLC columns to lower case.

    Args:
        symbol (str): The stock symbol to fetch data for.
        start_date (str): The start date for fetching data in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame: DataFrame with OHLC columns in lower case.
    """
    try:
        df = pdr.get_data_yahoo(symbol, start=start_date)
        df.index = pd.to_datetime(df.index)

        df.columns = [col.lower() for col in df.columns]
        df["close"] = df["adj close"]

        return df
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()


@tool(args_schema=StockStatsInput)
def get_stock_price_history(symbol: str) -> str:
    """Fetch a Stock's Price History by Symbol."""

    try:
        start_date = (datetime.now() - timedelta(days=365 * 2)).strftime("%Y-%m-%d")
        df = fetch_and_convert_ohlc(symbol, start_date)

        if df.empty:
            return (
                "\n<observation>\nNo data found for the given symbol\n</observation>\n"
            )

        df = add_technicals(df)
        df = df[-30:][::-1]

        return wrap_dataframe(df)
    except Exception as e:
        return f"\n<observation>\nError: {e}\n</observation>\n"


@tool(args_schema=StockStatsInput)
def get_stock_quantstats(symbol: str) -> str:
    """Fetch a Stock's Portfolio Analytics For Quants by Symbol."""

    try:
        start_date = (datetime.now() - timedelta(days=365 * 2)).strftime("%Y-%m-%d")
        df = fetch_and_convert_ohlc(symbol, start_date)

        if df.empty:
            return f"\n<observation>\nNo data found for the given symbol {symbol}\n</observation>\n"

        stock_ret = qs.utils.download_returns(symbol, period=df.index)
        bench_ret = qs.utils.download_returns("^GSPC", period=df.index)
        stats = qs.reports.metrics(
            stock_ret, mode="full", benchmark=bench_ret, display=False
        )

        return f"\n<observation>\n{stats}\n</observation>\n"
    except Exception as e:
        return f"\n<observation>\nError: {e}\n</observation>\n"


@tool
def get_gainers() -> str:
    """Fetch Top Price Gainers in the Stock Market."""

    try:
        gainers = obb.equity.discovery.gainers(sort="desc").to_df()

        if gainers.empty:
            return "\n<observation>\nNo gainers found\n</observation>\n"

        return wrap_dataframe(gainers)
    except Exception as e:
        return f"\n<observation>\nError: {e}\n</observation>\n"


@tool
def get_losers() -> str:
    """Fetch Stock Market's Top Losers."""

    try:
        losers = obb.equity.discovery.losers(sort="desc").to_df()

        if losers.empty:
            return "\n<observation>\nNo losers found\n</observation>\n"

        return wrap_dataframe(losers)
    except Exception as e:
        return f"\n<observation>\nError: {e}\n</observation>\n"


@tool(args_schema=StockStatsInput)
def get_stock_ratios(symbol: str) -> str:
    """Fetch an Extensive Set of Financial and Accounting Ratios for a Given Company Over Time."""

    try:
        trades = obb.equity.fundamental.ratios(symbol=symbol).to_df()

        if trades.empty:
            return f"\n<observation>\nNo data found for the given symbol {symbol}\n</observation>\n"

        return wrap_dataframe(trades)
    except Exception as e:
        return f"\n<observation>\nError: {e}\n</observation>\n"


@tool(args_schema=StockStatsInput)
def get_key_metrics(symbol: str) -> str:
    """Fetch Fundamental Metrics by Symbol."""

    try:
        metrics = obb.equity.fundamental.metrics(
            symbol=symbol, with_ttm=True, provider="yfinance"
        ).to_df()

        if metrics.empty:
            return f"\n<observation>\nNo data found for the given symbol {symbol}\n</observation>\n"

        return wrap_dataframe(metrics[::-1])
    except Exception as e:
        return f"\n<observation>\nError: {e}\n</observation>\n"


@tool(args_schema=StockStatsInput)
def get_stock_sector_info(symbol: str) -> str:
    """Fetch a Company's General Information By Symbol. This includes company name, industry, and sector data."""

    try:
        profile = obb.equity.profile(symbol=symbol).to_df()

        if profile.empty:
            return f"\n<observation>\nNo data found for the given symbol {symbol}\n</observation>\n"

        return wrap_dataframe(profile)
    except Exception as e:
        return f"\n<observation>\nError: {e}\n</observation>\n"


@tool(args_schema=StockStatsInput)
def get_valuation_multiples(symbol: str) -> str:
    """Fetch a Company's Valuation Multiples by Symbol."""

    try:
        df = obb.equity.fundamental.multiples(symbol=symbol).to_df()

        if df.empty:
            return f"\n<observation>\nNo data found for the given symbol {symbol}\n</observation>\n"

        return wrap_dataframe(df)
    except Exception as e:
        return f"\n<observation>\nError: {e}\n</observation>\n"


@tool
def get_stock_universe() -> str:
    """Fetch Bullish Trending Stocks Universe from FinViz."""

    try:
        return wrap_dataframe(fetch_custom_universe())
    except Exception as e:
        return f"\n<observation>\nError: {e}\n</observation>\n"
