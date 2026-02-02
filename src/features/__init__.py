import pandas as pd


def add_lag_returns(df: pd.DataFrame, lags=(1, 3, 5)) -> pd.DataFrame:
    """
    Add lagged return features.
    """
    for lag in lags:
        df[f"Return_Lag_{lag}"] = (
            df.groupby("Ticker")["Close"].pct_change(lag)
        )
    return df


def add_rolling_volatility(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Add rolling volatility feature.
    """
    df["Rolling_Volatility"] = (
        df.groupby("Ticker")["Close"]
        .pct_change()
        .rolling(window)
        .std()
    )
    return df


def add_volume_change(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add daily volume change feature.
    """
    df["Volume_Change"] = (
        df.groupby("Ticker")["Volume"].pct_change()
    )
    return df
