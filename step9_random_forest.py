# step9_random_forest.py (Corrected)

import numpy as np
from numba import cuda
import math
import scipy.sparse
from sklearn.model_selection import train_test_split
from collections import Counter

# --- CUDA Kernel (No changes here) ---

@cuda.jit
def find_best_split_kernel(data, labels, node_indices, n_features_subset, feature_indices,
                           min_impurity, best_feature, best_threshold):
    thread_idx = cuda.grid(1)
    
    feature_to_test_idx = thread_idx % n_features_subset
    if feature_to_test_idx >= feature_indices.shape[0]: return
    feature_col = feature_indices[feature_to_test_idx]

    sample_to_test_idx = thread_idx // n_features_subset
    
    if sample_to_test_idx < node_indices.shape[0]:
        threshold_idx = node_indices[sample_to_test_idx]
        threshold = data[threshold_idx, feature_col]

        left_counts = cuda.local.array(shape=3, dtype=np.int32)
        right_counts = cuda.local.array(shape=3, dtype=np.int32)
        for i in range(3):
            left_counts[i], right_counts[i] = 0, 0
            
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
        for i in range(3):
            p = left_counts[i] / n_left
            gini_left -= p * p
            
        gini_right = 1.0
        for i in range(3):
            p = right_counts[i] / n_right
            gini_right -= p * p

        total_samples = n_left + n_right
        weighted_gini = (n_left / total_samples) * gini_left + (n_right / total_samples) * gini_right

        old_min = cuda.atomic.min(min_impurity, 0, weighted_gini)
        
        if weighted_gini < old_min:
            best_feature[0] = feature_col
            best_threshold[0] = threshold

# --- CPU Helper Classes ---

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature, self.threshold, self.left, self.right, self.value = feature, threshold, left, right, value
    def is_leaf_node(self): return self.value is not None

class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=10, n_feats=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.root = None

    def fit(self, X, y):
        self.d_X = cuda.to_device(X.astype(np.float32))
        self.d_y = cuda.to_device(y.astype(np.int32))
        self.n_feats = X.shape[1] if not self.n_feats else min(self.n_feats, X.shape[1])
        self.root = self._grow_tree(X, y, np.arange(X.shape[0]), depth=0)

    def _grow_tree(self, X, y, indices, depth):
        labels = y[indices]
        
        if (depth >= self.max_depth or
            len(np.unique(labels)) == 1 or
            len(indices) < self.min_samples_split):
            leaf_value = Counter(labels).most_common(1)[0][0]
            return Node(value=leaf_value)

        feature, threshold = self._best_split_cuda(X, indices)

        if feature is None:
            leaf_value = Counter(labels).most_common(1)[0][0]
            return Node(value=leaf_value)
            
        left_indices_mask = X[indices, feature] <= threshold
        right_indices_mask = ~left_indices_mask
        left_indices = indices[left_indices_mask]
        right_indices = indices[right_indices_mask]
        
        left = self._grow_tree(X, y, left_indices, depth + 1)
        right = self._grow_tree(X, y, right_indices, depth + 1)
        
        return Node(feature, threshold, left, right)

    def _best_split_cuda(self, X, indices):
        feat_indices = np.random.choice(X.shape[1], self.n_feats, replace=False)
        
        d_node_indices = cuda.to_device(indices.astype(np.int32))
        d_feat_indices = cuda.to_device(feat_indices.astype(np.int32))
        
        d_min_impurity = cuda.to_device(np.array([1.1], dtype=np.float32))
        d_best_feature = cuda.to_device(np.array([-1], dtype=np.int32))
        d_best_threshold = cuda.to_device(np.array([-1.0], dtype=np.float32))
        
        threads_per_block = 256
        num_potential_splits = self.n_feats * len(indices)
        blocks_per_grid = (num_potential_splits + threads_per_block - 1) // threads_per_block
        
        if blocks_per_grid > 0:
            find_best_split_kernel[blocks_per_grid, threads_per_block](
                self.d_X, self.d_y, d_node_indices, self.n_feats, d_feat_indices,
                d_min_impurity, d_best_feature, d_best_threshold
            )
        
        best_feature_val = d_best_feature.copy_to_host()[0]
        best_threshold_val = d_best_threshold.copy_to_host()[0]

        if best_feature_val == -1: return None, None
        return best_feature_val, best_threshold_val

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node(): return node.value
        if x[node.feature] <= node.threshold: return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

class RandomForest:
    def __init__(self, n_trees=10, min_samples_split=2, max_depth=10, n_feats=None):
        self.n_trees, self.min_samples_split, self.max_depth, self.n_feats = n_trees, min_samples_split, max_depth, n_feats
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for i in range(self.n_trees):
            print(f"Training Tree {i+1}/{self.n_trees}...")
            tree = DecisionTree(min_samples_split=self.min_samples_split,
                                max_depth=self.max_depth, n_feats=self.n_feats)
            indices = np.random.choice(len(X), len(X), replace=True)
            X_sample, y_sample = X[indices], y[indices]
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        y_pred = [Counter(pred).most_common(1)[0][0] for pred in tree_preds]
        return np.array(y_pred)

def run_random_forest():
    print("--- Running Step 9: Random Forest (CUDA-Accelerated) ---")
    
    X_sparse = scipy.sparse.load_npz('preprocessed_features.npz')
    y = np.load('preprocessed_labels.npy')
    X = X_sparse.toarray()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    n_features = X.shape[1]
    n_feats_for_tree = int(np.sqrt(n_features))
    
    forest = RandomForest(n_trees=5, max_depth=10, n_feats=n_feats_for_tree)
    forest.fit(X_train, y_train)

    print("\nStarting prediction phase...")
    predictions = forest.predict(X_test)
    accuracy = np.mean(predictions == y_test)
    
    print(f"\nRandom Forest Final Accuracy: {accuracy * 100:.2f}%")
    print("------------------------------------------\n")

if __name__ == '__main__':
    run_random_forest()