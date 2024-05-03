from typing import List
from datetime import datetime, timedelta

from langchain.agents import tool

import pandas as pd
import numpy as np

from app.tools.utils import wrap_dataframe, fetch_stock_data, fetch_sp500_data
from app.tools.types import StockStatsInput


def calculate_performance(data: pd.DataFrame) -> float:
    """
    Calculate the performance of a stock over the given data period.
    """
    start_price = data["close"].iloc[0]
    end_price = data["close"].iloc[-1]
    performance = (end_price - start_price) / start_price
    return performance


def calculate_rs_rating(
    symbol: str, intervals: List[int], scaling_factor: int = 50
) -> pd.DataFrame:
    """
    Calculates the relative strength rating for a given stock symbol.

    The relative strength is based on the stock's performance compared to the S&P 500 index
    over the past 2 years, measured at the specified intervals (market sessions).
    """
    end_date = datetime.now()
    rs_ratings = []

    for interval in intervals:
        start_date = end_date - timedelta(days=interval)

        stock_data = fetch_stock_data(symbol, start_date, end_date)
        sp500_data = fetch_sp500_data(start_date, end_date)

        stock_data.index = pd.to_datetime(stock_data.index)
        sp500_data.index = pd.to_datetime(sp500_data.index)

        stock_performance = calculate_performance(stock_data)
        sp500_performance = calculate_performance(sp500_data)

        # Calculate relative performance to S&P 500 and apply scaling factor
        relative_performance = (stock_performance - sp500_performance) * scaling_factor

        # Normalize the relative performance to a 1-99 score
        # Assuming the distribution of relative performances is known and we aim for a midpoint of 50
        # This part may need adjustment based on a universe of empirical data
        scaled_score = np.clip(relative_performance + 50, 1, 99)

        rs_ratings.append(scaled_score)

    rs_df = pd.DataFrame({"Interval": intervals, "RS_Rating": rs_ratings})
    return rs_df


@tool(args_schema=StockStatsInput)
def get_relative_strength(symbol: str) -> str:
    """Calculate relative strength for a list of stocks."""

    session_intervals = [21, 63, 126, 189, 252]

    try:
        rs_rating = calculate_rs_rating(symbol, session_intervals)
        return wrap_dataframe(rs_rating)
    except Exception as e:
        return f"\n<observation>\nError: {e}\n</observation>\n"
