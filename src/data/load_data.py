import pandas as pd


def load_stock_data(filepath: str) -> pd.DataFrame:
    """
    Load and prepare stock price data.

    Steps:
    - Read CSV
    - Parse Date column (UTC)
    - Sort by Ticker and Date
    - Create Next-Day Return target

    Parameters
    ----------
    filepath : str
        Path to dataset CSV

    Returns
    -------
    pd.DataFrame
        Prepared stock dataframe
    """

    # Load dataset
    df = pd.read_csv(filepath)

    # Parse date and normalize timezone
    df["Date"] = pd.to_datetime(df["Date"], utc=True)

    # Sort correctly for time-series operations
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    # Target: next-day return
    df["Next_Day_Return"] = (
        df.groupby("Ticker")["Close"].shift(-1) - df["Close"]
    ) / df["Close"]

    return df
