import numpy as np
import scipy.sparse
from scipy.sparse.linalg import svds
import cupy as cp
from sklearn.model_selection import train_test_split
from collections import Counter
import time

# ---------- Corrected CUDA kernels ----------
find_best_split_kernel_code = r'''
extern "C" __global__
void find_best_split_kernel(const float* data, const int* labels, const int* node_indices,
                            int n_node_indices, int n_total_features,
                            int n_features_subset, const int* feature_indices,
                            float* best_impurities, int* best_features, float* best_thresholds,
                            int n_thresholds_per_feature) {

    // Shared memory layout (provided by launch): first blockDim.x floats for impurities,
    // next blockDim.x floats for thresholds.
    extern __shared__ float s_data[]; 
    float* s_impurities = s_data;
    float* s_thresholds = &s_data[blockDim.x];

    int tid = threadIdx.x;
    int feature_idx = blockIdx.x;
    if (feature_idx >= n_features_subset) return;

    int feature_col = feature_indices[feature_idx];

    // Each thread maintains a local best
    float best_local_impurity = 1.1f;
    float best_local_threshold = -1.0f;

    // Stride over candidate thresholds (we use the feature values at node_indices as candidates)
    for (int tidx = tid; tidx < n_node_indices; tidx += blockDim.x) {
        int sample_idx_for_thresh = node_indices[tidx];
        float threshold = data[sample_idx_for_thresh * n_total_features + feature_col];

        // Local counts for 3 classes (assumes labels in {0,1,2})
        int local_left[3] = {0,0,0};
        int local_right[3] = {0,0,0};
        int n_left = 0, n_right = 0;

        // Scan all node samples and partition by threshold
        for (int j = 0; j < n_node_indices; ++j) {
            int idx = node_indices[j];
            int lab = labels[idx];
            float val = data[idx * n_total_features + feature_col];
            if (val <= threshold) {
                local_left[lab]++;
                n_left++;
            } else {
                local_right[lab]++;
                n_right++;
            }
        }

        if (n_left > 0 && n_right > 0) {
            float gini_left = 1.0f, gini_right = 1.0f;
            for (int c = 0; c < 3; ++c) {
                float p_left = (float)local_left[c] / (float)n_left;
                float p_right = (float)local_right[c] / (float)n_right;
                gini_left -= p_left * p_left;
                gini_right -= p_right * p_right;
            }
            float weighted_gini = ((float)n_left / (float)n_node_indices) * gini_left +
                                  ((float)n_right / (float)n_node_indices) * gini_right;
            if (weighted_gini < best_local_impurity) {
                best_local_impurity = weighted_gini;
                best_local_threshold = threshold;
            }
        }
    }

    // Store local best into shared arrays
    s_impurities[tid] = best_local_impurity;
    s_thresholds[tid] = best_local_threshold;
    __syncthreads();

    // Parallel reduction to find the best among threads for this block
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (s_impurities[tid + s] < s_impurities[tid]) {
                s_impurities[tid] = s_impurities[tid + s];
                s_thresholds[tid] = s_thresholds[tid + s];
            }
        }
        __syncthreads();
    }

    // Thread 0 writes final result for this feature
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

    int votes0 = 0, votes1 = 0, votes2 = 0;

    for (int i = 0; i < n_stumps; ++i) {
        int feature = stump_features[i];
        float threshold = stump_thresholds[i];
        float val = data[sample_idx * n_features + feature];
        int pred = (val <= threshold) ? stump_left_vals[i] : stump_right_vals[i];
        if (pred == 0) votes0++;
        else if (pred == 1) votes1++;
        else if (pred == 2) votes2++;
    }

    // Find max vote
    int pred_final = 0;
    int max_vote = votes0;
    if (votes1 > max_vote) { max_vote = votes1; pred_final = 1; }
    if (votes2 > max_vote) { pred_final = 2; }

    predictions[sample_idx] = pred_final;
}
'''

find_best_split_kernel = cp.RawKernel(find_best_split_kernel_code, 'find_best_split_kernel')
predict_kernel = cp.RawKernel(predict_kernel_code, 'predict_kernel')

def run_stump_forest():
    print("--- Running Step 9: Optimized Random Forest with SVD ---")

    start_total = time.time()

    # 1) Load
    print("\n[1/5] Loading data...")
    start_load = time.time()
    X_sparse = scipy.sparse.load_npz('preprocessed_features.npz')
    y = np.load('preprocessed_labels.npy')
    print(f"    Original shape: {X_sparse.shape}")
    print(f"    Data loading time: {time.time() - start_load:.3f}s")

    # 2) SVD
    print("\n[2/5] Applying SVD dimensionality reduction...")
    start_svd = time.time()
    n_components = min(300, min(X_sparse.shape) - 1)
    print(f"    Reducing to {n_components} dimensions...")
    U, S, Vt = svds(X_sparse, k=n_components)
    X_reduced = U @ np.diag(S)
    print(f"    Reduced shape: {X_reduced.shape}")
    print(f"    SVD time: {time.time() - start_svd:.3f}s")

    # 3) Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_reduced, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4) Transfer to GPU
    print("\n[3/5] Transferring data to GPU...")
    start_transfer = time.time()
    d_X_train = cp.asarray(X_train.astype(np.float32))
    d_y_train = cp.asarray(y_train.astype(np.int32))
    d_X_test = cp.asarray(X_test.astype(np.float32))
    cp.cuda.Stream.null.synchronize()
    print(f"    GPU transfer time: {time.time() - start_transfer:.3f}s")

    n_trees = 10
    n_features = X_train.shape[1]
    n_feats_sqrt = max(1, int(np.sqrt(n_features)))

    stump_features = []
    stump_thresholds = []
    stump_left_vals = []
    stump_right_vals = []

    print(f"\n[4/5] Training {n_trees} trees...")
    start_train = time.time()
    threads_per_block = 256
    for i in range(n_trees):
        if (i + 1) % 5 == 0 or i == 0:
            print(f"    Training tree {i+1}/{n_trees}...")

        # Bootstrap sample (indices on CPU, then use to index d_X_train via fancy indexing on GPU)
        indices = np.random.choice(len(X_train), len(X_train), replace=True)
        d_X_sample = d_X_train[indices]
        d_y_sample = d_y_train[indices]

        # Random feature subset (CPU)
        feat_indices = np.random.choice(n_features, n_feats_sqrt, replace=False).astype(np.int32)
        d_feat_indices = cp.asarray(feat_indices)

        # node indices (we use 0..n-1 of the sample)
        n_node_indices = len(indices)
        d_node_indices = cp.arange(n_node_indices, dtype=cp.int32)

        # Output arrays (one entry per tested feature)
        d_best_impurities = cp.full(n_feats_sqrt, 1.1, dtype=cp.float32)
        d_best_features = cp.full(n_feats_sqrt, -1, dtype=cp.int32)
        d_best_thresholds = cp.full(n_feats_sqrt, -1.0, dtype=cp.float32)

        # Launch kernel: one block per feature; shared mem size = threads_per_block * 2 * 4 bytes (floats)
        blocks = n_feats_sqrt
        shared_mem_size = threads_per_block * 2 * 4  # two float arrays of length threads_per_block

        # kernel args: must match signature order and dtypes
        find_best_split_kernel(
            (blocks,), (threads_per_block,),
            (d_X_sample, d_y_sample, d_node_indices,
             np.int32(n_node_indices), np.int32(n_features),
             np.int32(n_feats_sqrt), d_feat_indices,
             d_best_impurities, d_best_features, d_best_thresholds, np.int32(n_node_indices)),
            shared_mem=shared_mem_size
        )

        # Bring results back to CPU to decide best feature
        impurities = d_best_impurities.get()
        best_idx = int(np.argmin(impurities))
        if impurities[best_idx] < 1.1:
            best_feat = int(d_best_features[best_idx].get())
            best_thresh = float(d_best_thresholds[best_idx].get())

            # compute leaf values on CPU (small arrays)
            X_sample_cpu = d_X_sample.get()
            y_sample_cpu = d_y_sample.get()

            left_mask = X_sample_cpu[:, best_feat] <= best_thresh
            right_mask = ~left_mask

            if np.any(left_mask):
                left_val = Counter(y_sample_cpu[left_mask]).most_common(1)[0][0]
            else:
                left_val = 0
            if np.any(right_mask):
                right_val = Counter(y_sample_cpu[right_mask]).most_common(1)[0][0]
            else:
                right_val = 0

            stump_features.append(best_feat)
            stump_thresholds.append(best_thresh)
            stump_left_vals.append(int(left_val))
            stump_right_vals.append(int(right_val))

    cp.cuda.Stream.null.synchronize()
    print(f"    Training time: {time.time() - start_train:.3f}s")
    print(f"    Trained {len(stump_features)} valid stumps")

    if len(stump_features) == 0:
        print("No stumps were trained successfully. Exiting.")
        return

    # 5) Predict
    print("\n[5/5] Predicting on GPU...")
    start_pred = time.time()
    d_stump_features = cp.asarray(stump_features, dtype=cp.int32)
    d_stump_thresholds = cp.asarray(stump_thresholds, dtype=cp.float32)
    d_stump_left = cp.asarray(stump_left_vals, dtype=cp.int32)
    d_stump_right = cp.asarray(stump_right_vals, dtype=cp.int32)
    d_predictions = cp.zeros(len(X_test), dtype=cp.int32)

    threads_per_block = 256
    blocks = (len(X_test) + threads_per_block - 1) // threads_per_block

    predict_kernel(
        (blocks,), (threads_per_block,),
        (d_X_test, np.int32(len(X_test)), np.int32(n_features),
         d_stump_features, d_stump_thresholds,
         d_stump_left, d_stump_right,
         np.int32(len(stump_features)), d_predictions)
    )

    cp.cuda.Stream.null.synchronize()
    final_preds = d_predictions.get()
    print(f"    Prediction time: {time.time() - start_pred:.3f}s")

    accuracy = np.mean(final_preds == y_test)

    print(f"\n{'='*50}")
    print(f"SVD-based Random Forest Results:")
    print(f"  - Dimensions: {X_sparse.shape[1]} → {n_components}")
    print(f"  - Accuracy: {accuracy * 100:.2f}%")
    print(f"  - Total Runtime: {time.time() - start_total:.3f}s")
    print(f"{'='*50}\n")


if __name__ == '__main__':
    run_stump_forest()
