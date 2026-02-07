import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from src.data.load_data import load_raw_data, preprocess_data, add_next_day_return
from src.features.build_features import (
    add_lag_returns,
    add_rolling_volatility,
    add_volume_change
)


DATA_PATH = "data/raw/Dataset.csv"


def prepare_dataset():
    df = load_raw_data(DATA_PATH)
    df = preprocess_data(df)
    df = add_next_day_return(df)

    df = add_lag_returns(df)
    df = add_rolling_volatility(df)
    df = add_volume_change(df)

    # Select features and target
    feature_cols = [
        "Return_Lag_1",
        "Return_Lag_3",
        "Return_Lag_5",
        "Rolling_Volatility",
        "Volume_Change",
    ]

    # Replace infinite values with NaN
    df = df.replace([float("inf"), float("-inf")], pd.NA)

    # Drop rows with missing or invalid values
    df = df.dropna(subset=feature_cols + ["Next_Day_Return"])


    X = df[feature_cols]
    y = df["Next_Day_Return"]

    return X, y


def train_baseline_model():
    X, y = prepare_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Baseline Linear Regression Results")
    print("--------------------------------")
    print(f"MSE : {mse:.6f}")
    print(f"R²  : {r2:.6f}")


if __name__ == "__main__":
    train_baseline_model()
