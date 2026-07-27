import pandas as pd
import sqlite3

df = pd.read_csv("Walmart_Sales.csv")
conn = sqlite3.connect("sample.db")

df.to_sql("sales", conn, if_exists="replace", index=False)

conn.close()

print(f"Loaded {len(df)} rows into sample.db")