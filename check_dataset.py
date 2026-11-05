import pandas as pd

df = pd.read_csv("temperature.csv", sep="\t")

print(df.head())
print(df.columns)
print(df.shape)