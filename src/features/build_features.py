import numpy as np
import pandas as pd


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering for stock return prediction
    """

    df = df.copy()

    # Daily return
    df["Return"] = df.groupby("Ticker")["Close"].pct_change()

    # Lagged returns
    df["Return_Lag_1"] = df.groupby("Ticker")["Return"].shift(1)
    df["Return_Lag_3"] = df.groupby("Ticker")["Return"].shift(3)
    df["Return_Lag_5"] = df.groupby("Ticker")["Return"].shift(5)

    # Rolling volatility (5-day)
    df["Rolling_Volatility"] = (
        df.groupby("Ticker")["Return"]
        .rolling(window=5)
        .std()
        .reset_index(level=0, drop=True)
    )

    # Volume change
    df["Volume_Change"] = df.groupby("Ticker")["Volume"].pct_change()

    return df
