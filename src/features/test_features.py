from src.data.load_data import load_raw_data, preprocess_data, add_next_day_return
from src.features.build_features import (
    add_lag_returns,
    add_rolling_volatility,
    add_volume_change
)

DATA_PATH = "data/raw/Dataset.csv"

df = load_raw_data(DATA_PATH)
df = preprocess_data(df)
df = add_next_day_return(df)

df = add_lag_returns(df)
df = add_rolling_volatility(df)
df = add_volume_change(df)

print(df.head())
print("\nFeature columns:")
print([col for col in df.columns if "Return" in col or "Volatility" in col or "Volume" in col])
