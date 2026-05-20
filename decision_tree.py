import numpy as np
from sklearn.datasets import load_diabetes
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

class Node:
    def __init__(self, feature = None, threshold = None, left = None, right = None, value = None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTree:
    def __init__(self, max_depth = 5, min_samples_split = 2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        self.root = self._build_tree(X, y)

    def _build_tree(self, X, y, depth = 0):
        samples, features = X.shape

        if depth > self.max_depth or self.min_samples_split > samples or np.var(y) == 0:
          leaf_value = np.mean(y)
          return Node(value = leaf_value)

        best_feature , best_threshold = self._best_split(X, y)

        if best_feature == None:
          return Node(value = np.mean(y))

        left_idx = X[:, best_feature] <= best_threshold
        right_idx = X[:, best_feature] > best_threshold

        left = self._build_tree(X[left_idx], y[left_idx], depth + 1)
        right = self._build_tree(X[right_idx], y[right_idx], depth + 1)

        return Node(best_feature, best_threshold, left, right)


    def _best_split(self, X, y):
      best_feature = None
      best_threshold = None
      best_mse = float('inf')

      samples, features = X.shape

      for feature in range(features):
        thresholds = np.unique(X[:, feature])

        for t in thresholds:
          left = X[:, feature] <= t
          right = X[:, feature] > t

          if len(y[left]) == 0 or len(y[right]) == 0:
            continue

          mse = self._compute_mse(y[left], y[right])

          if mse < best_mse:
            best_feature = feature
            best_threshold = t
            best_mse = mse

      return best_feature, best_threshold


    def _compute_mse(self, left, right):
      n = len(left) + len(right)

      return (len(left) / n * np.var(left) + len(right) / n * np.var(right))

    def _traverse_tree(self, x, node):
      if node.feature == None:
        return node.value

      if x[node.feature] <= node.threshold:
        return self._traverse_tree(x, node.left)
      else:
        return self._traverse_tree(x, node.right)

    def predict(self, X):
      return np.array([self._traverse_tree(x, self.root) for x in X])



regressor = DecisionTree(max_depth = 3)


data = load_diabetes()
df = pd.DataFrame(data.data, columns=data.feature_names)

#print(df.head())

X = df.iloc[:, :4].values
y = df.iloc[:, 4].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.99, random_state = 42)

regressor.fit(X_train, y_train)


pred = regressor.predict(X_test)

rmse = mean_squared_error(y_test, pred)

print(rmse)