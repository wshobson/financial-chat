import pandas_ta as ta


def add_technicals(df):
    df["last"] = df["close"]
    df["pct_change"] = df["close"].pct_change() * 100
    df["SMA_20"] = ta.sma(df["close"], length=20)
    df["SMA_50"] = ta.sma(df["close"], length=50)
    df["SMA_150"] = ta.sma(df["close"], length=150)
    df["SMA_200"] = ta.sma(df["close"], length=200)

    return df
