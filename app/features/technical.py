import pandas_ta as ta
import numpy as np
from sklearn.linear_model import LinearRegression


def detect_trendline(df):
    X = np.array(range(len(df))).reshape((-1, 1))
    y = df["close"].values.reshape((-1, 1))

    model = LinearRegression()
    model.fit(X, y)
    slope = model.coef_[0][0]

    return slope


def add_technicals(df):
    df["pct_change"] = df["close"].pct_change() * 100
    df["SMA_20"] = ta.sma(df["close"], length=20)
    df["SMA_50"] = ta.sma(df["close"], length=50)
    df["SMA_150"] = ta.sma(df["close"], length=150)
    df["SMA_200"] = ta.sma(df["close"], length=200)
    df["ATR"] = ta.atr(df["high"], df["low"], df["close"], length=14)
    df["RSI"] = ta.rsi(df["close"], length=14)
    df["52_WK_HIGH"] = df["close"].rolling(window=252).max()
    df["52_WK_LOW"] = df["close"].rolling(window=252).min()

    daily_range = df["high"] - df["low"]
    adr = daily_range.rolling(window=20).mean()

    df["ADR"] = (adr / df["close"]) * 100
    df["ADR_PCT"] = df["ADR"].fillna(0)

    df["TRENDLINE_SLOPE"] = detect_trendline(df)

    return df
