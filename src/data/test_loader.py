from src.data.load_data import (
    load_raw_data,
    preprocess_data,
    add_next_day_return
)


DATA_PATH = "data/raw/Dataset.csv"

df = load_raw_data(DATA_PATH)
df = preprocess_data(df)
df = add_next_day_return(df)

print(df.head())
print("\nColumns:", df.columns)
