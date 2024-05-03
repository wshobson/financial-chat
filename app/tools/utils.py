from datetime import datetime

import pandas as pd
from openbb import obb


def wrap_dataframe(df: pd.DataFrame) -> str:
    df_string = df.to_markdown(index=False)
    return f"\n<observation>\n{df_string}\n</observation>\n"


def fetch_stock_data(
    symbol: str, start_date: datetime, end_date: datetime
) -> pd.DataFrame:
    return obb.equity.price.historical(
        symbol,
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
        provider="yfinance",
    ).to_df()


def fetch_sp500_data(start_date: datetime, end_date: datetime) -> pd.DataFrame:
    return obb.equity.price.historical(
        "^GSPC",
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
        provider="yfinance",
    ).to_df()
