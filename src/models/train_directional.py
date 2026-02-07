import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.data.load_data import load_stock_data
from src.features.build_features import build_features


def main():
    # Load data
    df = load_stock_data("data/raw/Dataset.csv")
    df = build_features(df)

    # Create directional target
    df["Direction"] = np.where(df["Next_Day_Return"] > 0, 1, 0)

    # Feature set
    features = [
        "Return_Lag_1",
        "Return_Lag_3",
        "Return_Lag_5",
        "Rolling_Volatility",
        "Volume_Change"
    ]

    X = df[features]
    y = df["Direction"]

    # Handle invalid values
    X = X.replace([np.inf, -np.inf], np.nan)
    valid_idx = X.notna().all(axis=1)

    X = X[valid_idx]
    y = y[valid_idx]

    # Train-test split (time-aware)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        shuffle=False
    )

    # Train Logistic Regression
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluation
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    main()
