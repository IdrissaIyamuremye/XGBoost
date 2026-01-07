from sklearn.datasets import fetch_california_housing, load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np 
from XGBRegressionTree import XGBRegressionTree
data = fetch_california_housing()
X, y = data.data, data.target 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, test_size=0.2
)
print(X_train)
# Initialize your regression tree
tree = XGBRegressionTree(n_estimator=1, max_depth=3, reg_lambda=1.0, prune_gamma=0.0)

# Fit the tree on the test data
tree.fit(X_test, y_test)

# Predict using the same dataset
y_pred = tree.predict(X_test)

# Compare predictions with true values
print("\nPredictions vs True Values (first 10):")
for i in range(10):
    print(f"y_true={y_test[i]:.2f}, y_pred={y_pred[i]:.2f}")