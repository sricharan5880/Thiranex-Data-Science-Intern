import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix


df = pd.read_csv("titanic.csv")

print(df.head())


df = df[["Survived", "Pclass", "Sex", "Age", "Fare"]]

df["Age"] = df["Age"].fillna(df["Age"].mean())

df["Sex"] = df["Sex"].map({
    "male": 0,
    "female": 1
})


X = df[["Pclass", "Sex", "Age", "Fare"]]
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)



model = LogisticRegression()

model.fit(X_train, y_train)



y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(round(accuracy * 100, 2), "%")


cm = confusion_matrix(y_test, y_pred)

sns.heatmap(
    cm,
    annot=True,
    fmt='d'
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show(block=True)

plt.show()
