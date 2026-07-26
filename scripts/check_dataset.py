import pandas as pd

df = pd.read_csv("data/processed/training_dataset.csv")

print(df.head())
print()
print(df["label"].value_counts())
print()
print(df.describe())