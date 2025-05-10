import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score
url = "https://raw.githubusercontent.com/naveendaggubati17/DECISION-TREE-IMPLEMENTATION/main/drug200.csv"
df = pd.read_csv(url)
print(df.head())
print(df.info())
print(df.describe())
X = df.drop("Drug", axis=1)  # Features
y = df["Drug"]  # Target variable
X = pd.get_dummies(X, drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
plt.figure(figsize=(15, 10))
tree.plot_tree(clf, filled=True, feature_names=X.columns, class_names=np.unique(y))
plt.title("Decision Tree Visualization for Drug Classification")
plt.show()
