import pandas as pd

train_df = pd.read_csv("data/samsum-train.csv")
print("Train shape:", train_df.shape)
print("\nColumns:", train_df.columns.tolist())
print("\nSample row:\n", train_df.iloc[0])