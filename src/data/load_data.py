import pandas as pd


def load_raw_data(filepath: str) -> pd.DataFrame:
    """
    Load raw stock price data from CSV.
    """
    df = pd.read_csv(filepath)
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and prepare data for return prediction.
    """
    # Convert Date column to datetime
    df["Date"] = pd.to_datetime(df["Date"], utc=True)

    # Sort by stock and date (VERY IMPORTANT)
    df = df.sort_values(by=["Ticker", "Date"])

    # Drop rows with missing essential values
    df = df.dropna(subset=["Close", "Ticker", "Date"])

    return df


def add_next_day_return(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create Next-Day Return as target variable.
    Return_t+1 = (Close_{t+1} - Close_t) / Close_t
    """
    df["Next_Day_Return"] = (
        df.groupby("Ticker")["Close"].shift(-1) - df["Close"]
    ) / df["Close"]

    # Remove last row of each stock (no next day available)
    df = df.dropna(subset=["Next_Day_Return"])

    return df
