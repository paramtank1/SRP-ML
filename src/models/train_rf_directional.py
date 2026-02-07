import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score

from src.data.load_data import load_stock_data
from src.features.build_features import build_features


def main():
    # Load and prepare data
    df = load_stock_data("data/raw/Dataset.csv")
    df = build_features(df)

    # Directional target
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

    # TimeSeries K-Fold
    tscv = TimeSeriesSplit(n_splits=5)
    accuracies = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)

        print(f"Fold {fold} Accuracy: {acc:.4f}")

    print("\nAverage RF CV Accuracy:", sum(accuracies) / len(accuracies))

    # Feature Importance (from last fold)
    print("\nFeature Importances:")
    for feature, importance in zip(features, model.feature_importances_):
        print(f"{feature}: {importance:.4f}")


if __name__ == "__main__":
    main()
