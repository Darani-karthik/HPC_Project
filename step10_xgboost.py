# step10_xgboost.py (Final, Ultra-Compact CUDA Version)

import numpy as np
import scipy.sparse
import cupy as cp
from sklearn.model_selection import train_test_split

# 1. THE ONE CUSTOM KERNEL: Find Best Regression Split (MSE)
find_best_split_mse_kernel_code = r'''
extern "C" __global__
void find_best_split_mse(const float* data, const float* residuals, const int* sample_indices,
                         int n_samples_in_node, int n_total_features,
                         const int* feature_indices, int n_features_subset,
                         float* min_mse, int* best_feature, float* best_threshold) {
    
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int feature_to_test_idx = thread_idx % n_features_subset;
    if (feature_to_test_idx >= feature_indices[n_features_subset-1]+1) return;

    int feature_col = feature_indices[feature_to_test_idx];
    int sample_to_test_idx = thread_idx / n_features_subset;

    if (sample_to_test_idx < n_samples_in_node) {
        int threshold_sample_idx = sample_indices[sample_to_test_idx];
        float threshold = data[threshold_sample_idx * n_total_features + feature_col];

        float left_sum = 0, left_sq_sum = 0; int n_left = 0;
        float right_sum = 0, right_sq_sum = 0; int n_right = 0;

        for (int i = 0; i < n_samples_in_node; ++i) {
            int sample_idx = sample_indices[i];
            float res = residuals[sample_idx];
            if (data[sample_idx * n_total_features + feature_col] <= threshold) {
                left_sum += res; left_sq_sum += res * res; n_left++;
            } else {
                right_sum += res; right_sq_sum += res * res; n_right++;
            }
        }

        if (n_left > 0 && n_right > 0) {
            float left_var = (left_sq_sum / n_left) - ((left_sum / n_left) * (left_sum / n_left));
            float right_var = (right_sq_sum / n_right) - ((right_sum / n_right) * (right_sum / n_right));
            float weighted_mse = (n_left * left_var + n_right * right_var) / (n_left + n_right);
            if (weighted_mse < atomicMin(min_mse, weighted_mse)) {
                best_feature[0] = feature_col;
                best_threshold[0] = threshold;
            }
        }
    }
}
'''
find_best_split_mse_kernel = cp.RawKernel(find_best_split_mse_kernel_code, 'find_best_split_mse')

# 2. Main Script
print("--- Running Step 10: XGBoost (Ultra-Compact CUDA Stumps) ---")

# Load data & set params
X, y = scipy.sparse.load_npz('preprocessed_features.npz').toarray(), np.load('preprocessed_labels.npy')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
n_estimators, lr = 10, 0.1
n_samples, n_features = X_train.shape
n_classes = len(np.unique(y))

# Move data to GPU
d_X_train = cp.asarray(X_train, dtype=cp.float32)
d_X_test = cp.asarray(X_test, dtype=cp.float32)

# One-vs-Rest boosting
all_class_stumps, initial_predictions = [], {}
for i in range(n_classes):
    print(f"Training booster for class {i}...")
    d_y_binary = cp.where(cp.asarray(y_train) == i, 1.0, 0.0).astype(cp.float32)
    sum_y = float(cp.sum(d_y_binary))
    initial_predictions[i] = np.log((sum_y) / (n_samples - sum_y)) if sum_y > 0 and sum_y < n_samples else 0.0
    d_current_preds = cp.full(n_samples, initial_predictions[i], dtype=cp.float32)

    class_stumps = []
    for _ in range(n_estimators):
        d_probs = 1 / (1 + cp.exp(-d_current_preds))
        d_residuals = d_y_binary - d_probs
        
        # Find best split for a new stump using our CUDA kernel
        d_min_mse, d_best_feat, d_best_thresh = cp.array([1e9],'f4'), cp.array([-1],'i4'), cp.array([-1.],'f4')
        d_sample_idx = cp.arange(n_samples, dtype=cp.int32)
        d_feature_idx = cp.arange(n_features, dtype=cp.int32)
        tpb, bpg = (256,), ((n_features * n_samples + 255) // 256,)

        find_best_split_mse_kernel(bpg, tpb, (d_X_train, d_residuals, d_sample_idx, n_samples,
            n_features, d_feature_idx, n_features, d_min_mse, d_best_feat, d_best_thresh))
        
        best_feat, best_thresh = int(d_best_feat.get()), float(d_best_thresh.get())
        
        if best_feat != -1:
            # Create stump rules and update predictions entirely on GPU
            d_left_mask = d_X_train[:, best_feat] <= best_thresh
            left_val = float(cp.mean(d_residuals[d_left_mask]))
            right_val = float(cp.mean(d_residuals[~d_left_mask]))
            class_stumps.append({'feat': best_feat, 'thresh': best_thresh, 'left': left_val, 'right': right_val})
            d_current_preds += lr * cp.where(d_left_mask, left_val, right_val)
    
    all_class_stumps.append(class_stumps)

# Prediction
print("\nPredicting...")
all_class_probs = []
for i in range(n_classes):
    d_preds = cp.full(len(d_X_test), initial_predictions[i], dtype=cp.float32)
    for stump in all_class_stumps[i]:
        d_preds += lr * cp.where(d_X_test[:, stump['feat']] <= stump['thresh'], stump['left'], stump['right'])
    all_class_probs.append(1 / (1 + cp.exp(-d_preds)))

final_preds = cp.argmax(cp.stack(all_class_probs, axis=1), axis=1).get()
accuracy = np.mean(final_preds == y_test)

print(f"\nGradient Boosting Final Accuracy: {accuracy * 100:.2f}%")
print("------------------------------------------\n")
