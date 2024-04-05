import pandas_ta as ta


def add_technicals(df):
    df["pct_change"] = df["close"].pct_change() * 100
    df["current_date"] = df.index.strftime("%Y-%m-%d")
    df["SMA_20"] = ta.sma(df["close"], length=20)
    df["SMA_50"] = ta.sma(df["close"], length=50)
    df["SMA_150"] = ta.sma(df["close"], length=150)
    df["SMA_200"] = ta.sma(df["close"], length=200)
    df["ATR"] = ta.atr(df["high"], df["low"], df["close"], length=14)
    df["RSI"] = ta.rsi(df["close"], length=14)
    df["52_WK_HIGH"] = df["close"].rolling(window=252).max()
    df["52_WK_LOW"] = df["close"].rolling(window=252).min()

    return df
