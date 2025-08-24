# step10_xgboost.py

import numpy as np
from numba import cuda
import math
import scipy.sparse
from sklearn.model_selection import train_test_split
from collections import Counter

# --- Re-using the same CUDA kernel and CPU classes from Random Forest ---
# (In a real project, you would import these from a shared utility file)

@cuda.jit
def find_best_split_kernel(data, labels, node_indices, n_features_subset, feature_indices,
                           min_impurity, best_feature, best_threshold):
    thread_idx = cuda.grid(1)
    feature_to_test_idx = thread_idx % n_features_subset
    feature_col = feature_indices[feature_to_test_idx]
    sample_to_test_idx = thread_idx // n_features_subset
    
    if sample_to_test_idx < node_indices.shape[0]:
        threshold_idx = node_indices[sample_to_test_idx]
        threshold = data[threshold_idx, feature_col]

        left_counts = cuda.local.array(shape=3, dtype=np.int32)
        right_counts = cuda.local.array(shape=3, dtype=np.int32)
        for i in range(3): left_counts[i], right_counts[i] = 0, 0
        n_left, n_right = 0, 0

        for i in range(node_indices.shape[0]):
            sample_idx = node_indices[i]
            label = labels[sample_idx]
            if data[sample_idx, feature_col] <= threshold:
                left_counts[label] += 1
                n_left += 1
            else:
                right_counts[label] += 1
                n_right += 1

        if n_left == 0 or n_right == 0: return

        gini_left = 1.0
        for i in range(3): gini_left -= (left_counts[i] / n_left) ** 2
        gini_right = 1.0
        for i in range(3): gini_right -= (right_counts[i] / n_right) ** 2

        total_samples = n_left + n_right
        weighted_gini = (n_left / total_samples) * gini_left + (n_right / total_samples) * gini_right

        old_min = cuda.atomic.min(min_impurity, 0, weighted_gini)
        if weighted_gini < old_min:
            best_feature[0] = feature_col
            best_threshold[0] = threshold

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature, self.threshold, self.left, self.right, self.value = feature, threshold, left, right, value
    def is_leaf_node(self): return self.value is not None

class DecisionTreeRegressor: # Note: We need a REGRESSOR for XGBoost
    def __init__(self, min_samples_split=2, max_depth=3): # Shallow trees for boosting
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None
        # We don't use feature subsetting here for simplicity

    def fit(self, X, y):
        # For regressors, y are the residuals (continuous values)
        self.root = self._grow_tree(X, y, depth=0)

    def _grow_tree(self, X, y, depth):
        if (depth >= self.max_depth or len(np.unique(y)) == 1 or len(y) < self.min_samples_split):
            leaf_value = np.mean(y)
            return Node(value=leaf_value)
        
        feature, threshold = self._best_split(X, y)
        
        if feature is None:
            return Node(value=np.mean(y))
            
        left_indices = X[:, feature] <= threshold
        right_indices = X[:, feature] > threshold
        
        left = self._grow_tree(X[left_indices], y[left_indices], depth + 1)
        right = self._grow_tree(X[right_indices], y[right_indices], depth + 1)
        
        return Node(feature, threshold, left, right)

    def _best_split(self, X, y):
        # This is the CPU version of finding the best split for regression
        best_mse = float('inf')
        best_feat, best_thresh = None, None
        
        for feat_idx in range(X.shape[1]):
            thresholds = np.unique(X[:, feat_idx])
            for thresh in thresholds:
                left_indices = X[:, feat_idx] <= thresh
                y_left, y_right = y[left_indices], y[~left_indices]
                
                if len(y_left) > 0 and len(y_right) > 0:
                    mse = self._calculate_mse(y, y_left, y_right)
                    if mse < best_mse:
                        best_mse, best_feat, best_thresh = mse, feat_idx, thresh
                        
        return best_feat, best_thresh

    def _calculate_mse(self, y_parent, y_left, y_right):
        w_left, w_right = len(y_left) / len(y_parent), len(y_right) / len(y_parent)
        mse = w_left * np.var(y_left) + w_right * np.var(y_right)
        return mse

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node(): return node.value
        if x[node.feature] <= node.threshold: return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

# --- Main XGBoost Conceptual Class ---

class GradientBoosting:
    def __init__(self, n_estimators=10, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.initial_prediction = None

    def fit(self, X, y):
        # For multi-class, we need to do One-vs-Rest boosting
        self.n_classes = len(np.unique(y))
        
        for i in range(self.n_classes):
            print(f"Training booster for class {i} vs. the rest...")
            y_binary = np.where(y == i, 1.0, 0.0)
            
            # Start with an initial prediction (log-odds for classification)
            self.initial_prediction = np.log(np.sum(y_binary) / (len(y_binary) - np.sum(y_binary)))
            current_predictions = np.full(shape=y.shape, fill_value=self.initial_prediction)
            
            class_trees = []
            # --- THIS IS THE SEQUENTIAL PART THAT CANNOT BE PARALLELIZED ---
            for _ in range(self.n_estimators):
                # Calculate residuals (gradient of the loss function)
                # For logistic loss, it's (true_label - probability)
                probabilities = 1 / (1 + np.exp(-current_predictions))
                residuals = y_binary - probabilities
                
                # Train a new tree on the residuals
                tree = DecisionTreeRegressor(max_depth=self.max_depth)
                tree.fit(X, residuals)
                
                # Update the current predictions
                update = tree.predict(X)
                current_predictions += self.learning_rate * update
                
                class_trees.append(tree)
            self.trees.append(class_trees)
            
    def predict(self, X):
        all_class_probs = []
        for i in range(self.n_classes):
            # Start with the initial prediction
            predictions = np.full(shape=(X.shape[0],), fill_value=self.initial_prediction)
            # Add the predictions from each tree in the sequence
            for tree in self.trees[i]:
                predictions += self.learning_rate * tree.predict(X)
            
            # Convert log-odds back to probabilities
            probs = 1 / (1 + np.exp(-predictions))
            all_class_probs.append(probs)
            
        # Return the class with the highest probability
        return np.argmax(np.array(all_class_probs).T, axis=1)

# --- Main Function ---
def run_xgboost_demo():
    print("--- Running Step 10: XGBoost (Conceptual Demo) ---")
    print("NOTE: This demo highlights the sequential nature of boosting.")
    print("The individual trees are built on the CPU for clarity.")
    
    X_sparse = scipy.sparse.load_npz('preprocessed_features.npz')
    y = np.load('preprocessed_labels.npy')
    X = X_sparse.toarray()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Using a small number of estimators for a quick demonstration
    booster = GradientBoosting(n_estimators=10, max_depth=3, learning_rate=0.1)
    booster.fit(X_train, y_train)

    print("\nStarting prediction phase...")
    predictions = booster.predict(X_test)
    accuracy = np.mean(predictions == y_test)
    
    print(f"\nGradient Boosting Final Accuracy: {accuracy * 100:.2f}%")
    print("------------------------------------------\n")
    
if __name__ == '__main__':
    run_xgboost_demo()