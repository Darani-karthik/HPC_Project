# step9_random_forest.py 

import numpy as np
import scipy.sparse
import cupy as cp
from sklearn.model_selection import train_test_split
from collections import Counter

find_best_split_kernel_code = r'''
extern "C" _global_
void find_best_split_kernel(const float* data, const int* labels, const int* node_indices,
                            int n_node_indices, int n_total_features,
                            int n_features_subset, const int* feature_indices, int n_feature_indices,
                            float* min_impurity, int* best_feature, float* best_threshold) {
    
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int feature_to_test_idx = thread_idx % n_features_subset;
    if (feature_to_test_idx >= n_feature_indices) return;

    int feature_col = feature_indices[feature_to_test_idx];
    int sample_to_test_idx = thread_idx / n_features_subset;

    if (sample_to_test_idx < n_node_indices) {
        int threshold_idx = node_indices[sample_to_test_idx];
        float threshold = data[threshold_idx * n_total_features + feature_col];

        int left_counts[3] = {0}; int right_counts[3] = {0};
        int n_left = 0, n_right = 0;

        for (int i = 0; i < n_node_indices; ++i) {
            int sample_idx = node_indices[i];
            int label = labels[sample_idx];
            if (data[sample_idx * n_total_features + feature_col] <= threshold) {
                left_counts[label]++; n_left++;
            } else {
                right_counts[label]++; n_right++;
            }
        }

        if (n_left == 0 || n_right == 0) return;

        float gini_left = 1.0f, gini_right = 1.0f;
        for (int i = 0; i < 3; ++i) {
            gini_left -= ((float)left_counts[i] / n_left) * ((float)left_counts[i] / n_left);
            gini_right -= ((float)right_counts[i] / n_right) * ((float)right_counts[i] / n_right);
        }

        float weighted_gini = ((float)n_left / (n_left + n_right)) * gini_left + ((float)n_right / (n_left + n_right)) * gini_right;
        if (weighted_gini < atomicMin(min_impurity, weighted_gini)) {
            best_feature[0] = feature_col;
            best_threshold[0] = threshold;
        }
    }
}
'''
find_best_split_kernel = cp.RawKernel(find_best_split_kernel_code, 'find_best_split_kernel')

# 2. Main Script
def run_stump_forest():
    print("--- Running Step 9: Random Forest (Brutally Simplified Stumps) ---")
    
    X = scipy.sparse.load_npz('preprocessed_features.npz').toarray()
    y = np.load('preprocessed_labels.npy')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    n_trees = 10
    n_features = X.shape[1]
    n_feats_sqrt = int(np.sqrt(n_features))
    stumps = []

    for i in range(n_trees):
        print(f"Training Stump {i+1}/{n_trees}...")
        
        indices = np.random.choice(len(X_train), len(X_train), replace=True)
        X_sample, y_sample = X_train[indices], y_train[indices]

        d_X = cp.asarray(X_sample, dtype=cp.float32)
        d_y = cp.asarray(y_sample, dtype=cp.int32)
        
        feat_indices = np.random.choice(n_features, n_feats_sqrt, replace=False)
        d_node_indices = cp.arange(len(X_sample), dtype=cp.int32)
        d_feat_indices = cp.asarray(feat_indices, dtype=cp.int32)
        
        d_min_impurity = cp.array([1.1], dtype=cp.float32)
        d_best_feature = cp.array([-1], dtype=cp.int32)
        d_best_threshold = cp.array([-1.0], dtype=cp.float32)
        
        tpb = (256,)
        bpg = ((n_feats_sqrt * len(X_sample) + tpb[0] - 1) // tpb[0],)
        
        find_best_split_kernel(bpg, tpb, (d_X, d_y, d_node_indices, len(X_sample),
            n_features, n_feats_sqrt, d_feat_indices, len(feat_indices),
            d_min_impurity, d_best_feature, d_best_threshold))
        
        best_feat = int(d_best_feature.get()[0])
        best_thresh = d_best_threshold.get()[0]

        if best_feat != -1:
            left_mask = X_sample[:, best_feat] <= best_thresh
            right_mask = ~left_mask
            
            
            left_val = Counter(y_sample[left_mask]).most_common(1)[0][0] if np.any(left_mask) else Counter(y_sample).most_common(1)[0][0]
            right_val = Counter(y_sample[right_mask]).most_common(1)[0][0] if np.any(right_mask) else Counter(y_sample).most_common(1)[0][0]
            
            stumps.append({'feature': best_feat, 'threshold': best_thresh, 'left': left_val, 'right': right_val})

    print("\nPredicting...")
    all_preds = []
    for stump in stumps:
        preds = np.where(X_test[:, stump['feature']] <= stump['threshold'], stump['left'], stump['right'])
        all_preds.append(preds)
    
    final_preds = np.array([Counter(sample_preds).most_common(1)[0][0] for sample_preds in np.array(all_preds).T])
    accuracy = np.mean(final_preds == y_test)
    
    print(f"\nRandom Stump Forest Final Accuracy: {accuracy * 100:.2f}%")
    print("------------------------------------------\n")

if _name_ == '_main_':
    run_stump_forest()
