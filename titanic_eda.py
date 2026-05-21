import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("titanic.csv")

print("First 5 Rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())


print("\nMissing Values:")
print(df.isnull().sum())


sns.countplot(x="Survived", data=df)

plt.title("Survival Count")

plt.show()


sns.countplot(x="Sex", data=df)

plt.title("Gender Distribution")

plt.show()


plt.hist(df["Age"].dropna(), bins=20)

plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")

plt.show()



df["Sex"] = df["Sex"].map({
    "male": 0,
    "female": 1
})


corr_data = df[[
    "Survived",
    "Pclass",
    "Sex",
    "Age",
    "Fare"
]]


corr = corr_data.corr()


sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")

plt.show()


print("\nInsights:")
print("1. Female passengers survived more.")
print("2. Higher class passengers had better survival rates.")
print("3. Most passengers were between age 20-40.")
print("4. Fare and passenger class influence survival.")
