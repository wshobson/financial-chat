from datetime import date, timedelta

import pandas_ta as ta

from langchain.agents import tool

from app.tools.utils import fetch_stock_data
from app.tools.types import StockStatsInput, RMultipleInput, PositionSizingInput


def calculate_technical_levels(df):
    df["SMA_20"] = ta.sma(df["close"], length=20)
    df["SMA_50"] = ta.sma(df["close"], length=50)
    df["SMA_200"] = ta.sma(df["close"], length=200)

    support = df["low"].rolling(window=20).min().iloc[-1]
    resistance = df["high"].rolling(window=20).max().iloc[-1]

    return (
        support,
        resistance,
        df["SMA_20"].iloc[-1],
        df["SMA_50"].iloc[-1],
        df["SMA_200"].iloc[-1],
    )


@tool(args_schema=StockStatsInput)
def calculate_technical_stops(symbol: str) -> str:
    """Calculate stops at key technical levels for a given stock."""

    try:
        end_date = date.today()
        start_date = end_date - timedelta(days=365)

        df = fetch_stock_data(symbol, start_date, end_date)
        support, resistance, sma_20, sma_50, sma_200 = calculate_technical_levels(df)

        stop_levels = [
            ("Support", support),
            ("Resistance", resistance),
            ("20-day SMA", sma_20),
            ("50-day SMA", sma_50),
            ("200-day SMA", sma_200),
        ]

        stop_levels_str = "\n".join(
            [f"{level[0]}: {level[1]:.2f}" for level in stop_levels]
        )

        return f"\n<observation>\nPotential stop levels for {symbol}:\n{stop_levels_str}\n</observation>\n"
    except Exception as e:
        return f"\n<observation>\nError: {e}\n</observation>\n"


@tool(args_schema=RMultipleInput)
def calculate_r_multiples(
    symbol: str, entry_price: float, stop_price: float, risk_multiple: int = 2
) -> str:
    """Calculate potential profit targets and stop-losses based on R multiples."""

    try:
        initial_risk = abs(entry_price - stop_price)
        profit_target = entry_price + (initial_risk * risk_multiple)
        stop = entry_price - initial_risk

        return f"\n<observation>\nFor {symbol}:\nEntry Price: {entry_price:.2f}\nInitial Risk: {initial_risk:.2f}\nProfit Target ({risk_multiple}R): {profit_target:.2f}\nStop-Loss: {stop:.2f}\n</observation>\n"
    except Exception as e:
        return f"\n<observation>\nError: {e}\n</observation>\n"


@tool(args_schema=PositionSizingInput)
def calculate_position_size(
    symbol: str,
    entry_price: float,
    stop_price: float,
    account_size: float = 100000.00,
    risk_percent: float = 1.0,
) -> str:
    """Calculate the optimal position size for a trade based on account size and risk tolerance."""

    try:
        risk_amount = account_size * (risk_percent / 100)
        potential_loss = abs(entry_price - stop_price)
        position_size = risk_amount / potential_loss

        return f"\n<observation>\nOptimal position size for {symbol}: {position_size:.2f} shares\nEntry Price: {entry_price:.2f}\nStop Price: {stop_price:.2f}\nPotential Loss Per Share: {potential_loss:.2f}\n</observation>\n"
    except Exception as e:
        return f"\n<observation>\nError: {e}\n</observation>\n"
