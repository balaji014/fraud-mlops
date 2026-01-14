import pandas as pd

df = pd.read_csv("data/creditcard.csv")
print(df.columns.tolist()[:15])
