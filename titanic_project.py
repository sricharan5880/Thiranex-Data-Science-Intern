import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("titanic.csv")

print("Original Dataset:")
print(df.head())


print("\nMissing Values:")
print(df.isnull().sum())

df["Age"] = df["Age"].fillna(df["Age"].mean())

df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

df = df.drop_duplicates()


df = df.drop(columns=["Cabin"])

print("\nCleaned Dataset:")
print(df.head())


survived = df["Survived"].value_counts()

print("\nSurvival Count:")
print(survived)

print("\nAverage Age:")
print(df["Age"].mean())


plt.bar(["Not Survived", "Survived"], survived)

plt.title("Titanic Survival Count")
plt.xlabel("Status")
plt.ylabel("Passengers")

plt.show()

plt.hist(df["Age"], bins=20)

plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")

plt.show()
