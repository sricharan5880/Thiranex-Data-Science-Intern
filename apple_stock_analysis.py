import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("apple_stock.csv")

print("First 5 Rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

plt.figure(figsize=(10,5))

plt.plot(df["Date"], df["AAPL.Close"])

plt.title("Apple Closing Stock Price")
plt.xlabel("Date")
plt.ylabel("Closing Price")

plt.xticks(rotation=45)

plt.show()

plt.plot(df["Date"], df["AAPL.Volume"])

plt.title("Apple Trading Volume")
plt.xlabel("Date")
plt.ylabel("Volume")

plt.xticks(rotation=45)

plt.show()

plt.figure(figsize=(10,5))

plt.plot(df["Date"], df["AAPL.High"], label="High")
plt.plot(df["Date"], df["AAPL.Low"], label="Low")

plt.title("Apple High vs Low Prices")

plt.xlabel("Date")
plt.ylabel("Price")

plt.legend()

plt.xticks(rotation=45)

plt.show()

corr = df[[
    "AAPL.Open",
    "AAPL.High",
    "AAPL.Low",
    "AAPL.Close",
    "AAPL.Volume"
]].corr()

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")

plt.show()

print("\nInsights:")
print("1. Apple stock prices show an overall upward trend.")
print("2. High and low prices are strongly correlated.")
print("3. Trading volume changes significantly over time.")
print("4. Opening and closing prices are closely related.")
