import numpy as np
import scipy.sparse
import cupy as cp
from sklearn.model_selection import train_test_split
from collections import Counter
import time

# Optimized kernel with proper parallelization
find_best_split_kernel_code = r'''
extern "C" __global__
void find_best_split_kernel(const float* data, const int* labels, const int* node_indices,
                            int n_node_indices, int n_total_features,
                            int n_features_subset, const int* feature_indices,
                            float* best_impurities, int* best_features, float* best_thresholds,
                            int n_thresholds_per_feature) {
    
    extern __shared__ int shared_mem[];
    int* left_counts = shared_mem;  // 3 ints per feature
    int* right_counts = &shared_mem[n_features_subset * 3];  // 3 ints per feature
    
    int tid = threadIdx.x;
    int feature_idx = blockIdx.x;
    
    if (feature_idx >= n_features_subset) return;
    
    int feature_col = feature_indices[feature_idx];
    
    // Initialize shared memory
    if (tid < 3) {
        left_counts[feature_idx * 3 + tid] = 0;
        right_counts[feature_idx * 3 + tid] = 0;
    }
    __syncthreads();
    
    // Each block processes one feature, trying multiple thresholds
    float best_local_impurity = 1.1f;
    float best_local_threshold = -1.0f;
    
    // Stride through potential thresholds
    for (int thresh_idx = tid; thresh_idx < n_node_indices; thresh_idx += blockDim.x) {
        int sample_idx = node_indices[thresh_idx];
        float threshold = data[sample_idx * n_total_features + feature_col];
        
        // Count classes in left and right splits (local to this thread)
        int local_left[3] = {0, 0, 0};
        int local_right[3] = {0, 0, 0};
        int n_left = 0, n_right = 0;
        
        for (int i = 0; i < n_node_indices; ++i) {
            int idx = node_indices[i];
            int label = labels[idx];
            if (data[idx * n_total_features + feature_col] <= threshold) {
                local_left[label]++;
                n_left++;
            } else {
                local_right[label]++;
                n_right++;
            }
        }
        
        if (n_left > 0 && n_right > 0) {
            float gini_left = 1.0f, gini_right = 1.0f;
            for (int i = 0; i < 3; ++i) {
                float p_left = (float)local_left[i] / n_left;
                float p_right = (float)local_right[i] / n_right;
                gini_left -= p_left * p_left;
                gini_right -= p_right * p_right;
            }
            
            float weighted_gini = ((float)n_left / n_node_indices) * gini_left + 
                                  ((float)n_right / n_node_indices) * gini_right;
            
            if (weighted_gini < best_local_impurity) {
                best_local_impurity = weighted_gini;
                best_local_threshold = threshold;
            }
        }
    }
    
    // Reduction within block to find best split for this feature
    __shared__ float s_impurities[256];
    __shared__ float s_thresholds[256];
    
    s_impurities[tid] = best_local_impurity;
    s_thresholds[tid] = best_local_threshold;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && s_impurities[tid + s] < s_impurities[tid]) {
            s_impurities[tid] = s_impurities[tid + s];
            s_thresholds[tid] = s_thresholds[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        best_impurities[feature_idx] = s_impurities[0];
        best_features[feature_idx] = feature_col;
        best_thresholds[feature_idx] = s_thresholds[0];
    }
}
'''

predict_kernel_code = r'''
extern "C" __global__
void predict_kernel(const float* data, int n_samples, int n_features,
                   const int* stump_features, const float* stump_thresholds,
                   const int* stump_left_vals, const int* stump_right_vals,
                   int n_stumps, int* predictions) {
    
    int sample_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample_idx >= n_samples) return;
    
    int votes[3] = {0, 0, 0};
    
    for (int i = 0; i < n_stumps; ++i) {
        int feature = stump_features[i];
        float threshold = stump_thresholds[i];
        float val = data[sample_idx * n_features + feature];
        
        int pred = (val <= threshold) ? stump_left_vals[i] : stump_right_vals[i];
        votes[pred]++;
    }
    
    // Find max vote
    int max_vote = votes[0];
    int prediction = 0;
    for (int i = 1; i < 3; ++i) {
        if (votes[i] > max_vote) {
            max_vote = votes[i];
            prediction = i;
        }
    }
    
    predictions[sample_idx] = prediction;
}
'''

find_best_split_kernel = cp.RawKernel(find_best_split_kernel_code, 'find_best_split_kernel')
predict_kernel = cp.RawKernel(predict_kernel_code, 'predict_kernel')

def run_stump_forest():
    print("--- Running Step 9: Optimized Random Forest ---")
    
    # Start total timer
    start_total = time.time()
    
    # Data loading
    print("\n[1/4] Loading data...")
    start_load = time.time()
    X = scipy.sparse.load_npz('preprocessed_features.npz').toarray()
    y = np.load('preprocessed_labels.npy')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"    Data loading time: {time.time() - start_load:.3f}s")
    
    # GPU transfer
    print("\n[2/4] Transferring data to GPU...")
    start_transfer = time.time()
    d_X_train = cp.asarray(X_train, dtype=cp.float32)
    d_y_train = cp.asarray(y_train, dtype=cp.int32)
    d_X_test = cp.asarray(X_test, dtype=cp.float32)
    cp.cuda.Stream.null.synchronize()
    print(f"    GPU transfer time: {time.time() - start_transfer:.3f}s")
    
    n_trees = 10
    n_features = X_train.shape[1]
    n_feats_sqrt = int(np.sqrt(n_features))
    
    stump_features = []
    stump_thresholds = []
    stump_left_vals = []
    stump_right_vals = []

    print(f"\n[3/4] Training {n_trees} trees...")
    start_train = time.time()

    for i in range(n_trees):
        if (i + 1) % 5 == 0 or i == 0:
            print(f"    Training tree {i+1}/{n_trees}...")
        
        # Bootstrap sample
        indices = cp.random.choice(len(X_train), len(X_train), replace=True)
        d_X_sample = d_X_train[indices]
        d_y_sample = d_y_train[indices]
        
        # Random feature subset
        feat_indices = np.random.choice(n_features, n_feats_sqrt, replace=False)
        d_feat_indices = cp.asarray(feat_indices, dtype=cp.int32)
        d_node_indices = cp.arange(len(indices), dtype=cp.int32)
        
        # Output arrays (one per feature)
        d_best_impurities = cp.full(n_feats_sqrt, 1.1, dtype=cp.float32)
        d_best_features = cp.full(n_feats_sqrt, -1, dtype=cp.int32)
        d_best_thresholds = cp.full(n_feats_sqrt, -1.0, dtype=cp.float32)
        
        # Launch kernel: one block per feature
        threads_per_block = 256
        blocks = n_feats_sqrt
        shared_mem_size = n_feats_sqrt * 6 * 4  # (left_counts + right_counts) * 3 classes * 4 bytes
        
        find_best_split_kernel(
            (blocks,), (threads_per_block,),
            (d_X_sample, d_y_sample, d_node_indices, len(indices),
             n_features, n_feats_sqrt, d_feat_indices,
             d_best_impurities, d_best_features, d_best_thresholds, len(indices)),
            shared_mem=shared_mem_size
        )
        
        # Find best feature across all tested features (on CPU)
        impurities = d_best_impurities.get()
        best_idx = cp.argmin(d_best_impurities).get()
        
        if impurities[best_idx] < 1.1:
            best_feat = int(d_best_features[best_idx].get())
            best_thresh = float(d_best_thresholds[best_idx].get())
            
            # Compute leaf values
            X_sample_cpu = d_X_sample.get()
            y_sample_cpu = d_y_sample.get()
            
            left_mask = X_sample_cpu[:, best_feat] <= best_thresh
            right_mask = ~left_mask
            
            left_val = Counter(y_sample_cpu[left_mask]).most_common(1)[0][0] if np.any(left_mask) else 0
            right_val = Counter(y_sample_cpu[right_mask]).most_common(1)[0][0] if np.any(right_mask) else 0
            
            stump_features.append(best_feat)
            stump_thresholds.append(best_thresh)
            stump_left_vals.append(left_val)
            stump_right_vals.append(right_val)
    
    cp.cuda.Stream.null.synchronize()
    print(f"    Training time: {time.time() - start_train:.3f}s")

    print("\n[4/4] Predicting on GPU...")
    
    # Prepare stump data on GPU
    start_pred = time.time()
    d_stump_features = cp.asarray(stump_features, dtype=cp.int32)
    d_stump_thresholds = cp.asarray(stump_thresholds, dtype=cp.float32)
    d_stump_left = cp.asarray(stump_left_vals, dtype=cp.int32)
    d_stump_right = cp.asarray(stump_right_vals, dtype=cp.int32)
    d_predictions = cp.zeros(len(X_test), dtype=cp.int32)
    
    # Launch prediction kernel
    threads_per_block = 256
    blocks = (len(X_test) + threads_per_block - 1) // threads_per_block
    
    predict_kernel(
        (blocks,), (threads_per_block,),
        (d_X_test, len(X_test), n_features,
         d_stump_features, d_stump_thresholds,
         d_stump_left, d_stump_right,
         len(stump_features), d_predictions)
    )
    
    cp.cuda.Stream.null.synchronize()
    final_preds = d_predictions.get()
    print(f"    Prediction time: {time.time() - start_pred:.3f}s")
    
    accuracy = np.mean(final_preds == y_test)
    
    print(f"\n{'='*50}")
    print(f"Random Stump Forest Accuracy: {accuracy * 100:.2f}%")
    print(f"Total Runtime: {time.time() - start_total:.3f}s")
    print(f"{'='*50}\n")

if __name__ == '__main__':
    run_stump_forest()
