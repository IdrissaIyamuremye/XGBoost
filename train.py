from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np 
from DecisionTree import DecisionTree
data = datasets.load_breast_cancer()
X, y = data.data, data.target 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, test_size=0.2
)
cls = DecisionTree()
cls.fit(X_train, y_train)
predictions = cls.predict(X_test)
def accuracy(y_test, y_pred):
    return np.sum(y_test==y_pred) / len(y_test)
acc = accuracy(y_test, predictions)
print(f"Accuracry: {acc: .3f}")
