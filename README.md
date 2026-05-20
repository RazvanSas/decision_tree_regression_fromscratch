# Decision Tree Regressor from Scratch (Python)

## Overview
This project implements a **Decision Tree Regressor from scratch in Python**, without using machine learning libraries such as scikit-learn.

The goal of this project is to understand how decision trees work internally, including recursive splitting, impurity reduction, and tree-based prediction.

---

## Key Features
- Fully custom implementation of a regression decision tree
- Recursive tree construction using a `Node` class
- Greedy split search over all features and thresholds
- Variance-based split criterion (impurity reduction)
- Depth-limited tree growth to prevent overfitting
- Prediction using recursive tree traversal

---

## How the Algorithm Works

### 1. Tree Construction
The tree is built recursively:
- For each node, all possible feature-threshold splits are evaluated
- The best split is chosen based on **minimum variance (impurity)**
- Data is split into left and right subtrees
- Process repeats until stopping conditions are met

### 2. Stopping Conditions
The recursion stops when:
- Maximum depth is reached
- Number of samples is too small
- All target values are identical (zero variance)

### 3. Prediction
Prediction is done by:
- Traversing the tree from root to leaf
- Following decision rules at each node
- Returning the leaf node value (mean of targets)

---

## Model Details
- Model type: Regression Decision Tree
- Split criterion: Variance reduction (MSE-equivalent impurity)
- Tree structure: Binary recursive tree
- Leaf prediction: Mean of target values

---

## Dataset
The model was tested using the **Diabetes dataset** from `sklearn.datasets`.

---

## Technologies Used
- Python
- NumPy
- Pandas
- scikit-learn (only for dataset loading and evaluation)

---

## Evaluation Metrics
- Mean Squared Error (MSE)



