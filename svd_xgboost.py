# step10_xgboost_svd_integrated.py (Fixed: atomicMin for float via uint-bit trick)
import numpy as np
import scipy.sparse
import cupy as cp
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer # New import for preprocessing

print("--- Running Step 10: XGBoost (CUDA + SVD Embedding) on fin_data_1.csv ---")

# ============================================================== 
# 1. CUSTOM CUDA KERNEL: Find Best Regression Split (MSE)
#    Uses uint-bit reinterpretation to perform atomicMin on floats
# ============================================================== 
find_best_split_mse_kernel_code = r'''
extern "C" __global__
void find_best_split_mse(const float* data, const float* residuals, const int* sample_indices,
                         int n_samples_in_node, int n_total_features,
                         const int* feature_indices, int n_features_subset,
                         unsigned int* min_mse_bits, int* best_feature, float* best_threshold) {

    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = n_features_subset * n_samples_in_node;
    if (thread_idx >= total_threads) return;

    int feature_to_test_idx = thread_idx % n_features_subset;
    int sample_to_test_idx = thread_idx / n_features_subset;

    int feature_col = feature_indices[feature_to_test_idx];

    int threshold_sample_idx = sample_indices[sample_to_test_idx];
    float threshold = data[threshold_sample_idx * n_total_features + feature_col];

    float left_sum = 0.0f, left_sq_sum = 0.0f; int n_left = 0;
    float right_sum = 0.0f, right_sq_sum = 0.0f; int n_right = 0;

    for (int i = 0; i < n_samples_in_node; ++i) {
        int sample_idx = sample_indices[i];
        float res = residuals[sample_idx];
        float feat_val = data[sample_idx * n_total_features + feature_col];
        if (feat_val <= threshold) {
            left_sum += res; left_sq_sum += res * res; n_left++;
        } else {
            right_sum += res; right_sq_sum += res * res; n_right++;
        }
    }

    if (n_left > 0 && n_right > 0) {
        float left_var = (left_sq_sum / n_left) - ((left_sum / n_left) * (left_sum / n_left));
        float right_var = (right_sq_sum / n_right) - ((right_sum / n_right) * (right_sum / n_right));
        float weighted_mse = (n_left * left_var + n_right * right_var) / (n_left + n_right);

        // Reinterpret float bits as unsigned int for atomicMin (works for non-negative floats)
        unsigned int new_bits = __float_as_uint(weighted_mse);
        unsigned int old_bits = atomicMin(min_mse_bits, new_bits); // returns the previous value
        float old_val = __uint_as_float(old_bits);

        // If the old value was greater than this weighted_mse, this thread just wrote a better value
        if (old_val > weighted_mse) {
            // Update best feature & threshold (note: possible race between threads writing these;
            // but the atomicMin guaranteed this thread's weighted_mse was accepted as the new minimum)
            best_feature[0] = feature_col;
            best_threshold[0] = threshold;
        }
    }
}
'''
find_best_split_mse_kernel = cp.RawKernel(find_best_split_mse_kernel_code, 'find_best_split_mse')

# ============================================================== 
# 2. DATA LOADING + GPU SVD EMBEDDING
#    (Integrated Preprocessing from fin_data_1.csv)
# ============================================================== 
print("Loading fin_data_1.csv, preprocessing, and embedding data with SVD (on GPU)...")

# Load data
df = pd.read_csv('fin_data_1.csv')

# Map Sentiment to numerical labels (0, 1, 2)
sentiment_map = {'negative': 0, 'neutral': 1, 'positive': 2}
df['Sentiment_Numeric'] = df['Sentiment'].map(sentiment_map)
y = df['Sentiment_Numeric'].values

# TF-IDF Vectorization
# Limiting features to a manageable number (5000) for SVD/GPU memory
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000) 
X_sparse = vectorizer.fit_transform(df['Sentence'])

# Convert sparse matrix to dense array for CuPy/SVD operation
X = X_sparse.toarray().astype(np.float32)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Move to GPU for SVD computation
d_X_full = cp.asarray(X_train, dtype=cp.float32)

# Perform SVD using CuPy (on GPU)
k = min(100, d_X_full.shape[1])  # Reduce to 100 components or fewer
U, S, VT = cp.linalg.svd(d_X_full, full_matrices=False)

# Project data into reduced dimensionality
d_X_train = (U[:, :k] * S[:k])  # Low-dimensional embedding (n_samples x k)
d_X_test = cp.asarray(X_test, dtype=cp.float32).dot(VT[:k, :].T)

n_samples, n_features = d_X_train.shape
n_classes = len(np.unique(y))
n_estimators, lr = 10, 0.1

print(f"SVD embedding complete → Reduced dimensions: {n_features}")

# ============================================================== 
# 3. GRADIENT BOOSTING TRAINING (One-vs-Rest)
# ============================================================== 
all_class_stumps, initial_predictions = [], {}

for i in range(n_classes):
    print(f"Training booster for class {i}...")
    d_y_binary = cp.where(cp.asarray(y_train) == i, 1.0, 0.0).astype(cp.float32)
    sum_y = float(cp.sum(d_y_binary))
    initial_predictions[i] = np.log((sum_y) / (n_samples - sum_y)) if 0 < sum_y < n_samples else 0.0
    d_current_preds = cp.full(n_samples, initial_predictions[i], dtype=cp.float32)

    class_stumps = []
    for _ in range(n_estimators):
        d_probs = 1 / (1 + cp.exp(-d_current_preds))
        d_residuals = d_y_binary - d_probs

        # Prepare min_mse as uint32 containing the float bits of a large float (1e9)
        init_min_float = np.array([np.float32(1e9)], dtype=np.float32)
        init_min_bits = init_min_float.view(np.uint32)   # numpy uint32 array with same bits
        d_min_mse_bits = cp.asarray(init_min_bits, dtype=cp.uint32)

        d_best_feat = cp.array([-1], dtype=cp.int32)
        d_best_thresh = cp.array([-1.0], dtype=cp.float32)

        d_sample_idx = cp.arange(n_samples, dtype=cp.int32)
        d_feature_idx = cp.arange(n_features, dtype=cp.int32)

        total_threads = n_features * n_samples
        threads_per_block = 256
        blocks_per_grid = (total_threads + threads_per_block - 1) // threads_per_block

        find_best_split_mse_kernel((blocks_per_grid,), (threads_per_block,), (
            d_X_train, d_residuals, d_sample_idx, n_samples,
            n_features, d_feature_idx, n_features,
            d_min_mse_bits, d_best_feat, d_best_thresh
        ))

        best_feat, best_thresh = int(d_best_feat.get()), float(d_best_thresh.get())

        if best_feat != -1:
            # compute stump outputs using masks
            d_left_mask = d_X_train[:, best_feat] <= best_thresh
            # Avoid empty-slice mean warnings by guarding counts
            if int(cp.sum(d_left_mask)) == 0 or int(cp.sum(~d_left_mask)) == 0:
                # skip invalid split
                continue
            left_val = float(cp.mean(d_residuals[d_left_mask]))
            right_val = float(cp.mean(d_residuals[~d_left_mask]))
            class_stumps.append({'feat': best_feat, 'thresh': best_thresh, 'left': left_val, 'right': right_val})
            d_current_preds += lr * cp.where(d_left_mask, left_val, right_val)

    all_class_stumps.append(class_stumps)

# ============================================================== 
# 4. EVALUATION
# ============================================================== 
print("\nPredicting...")
all_class_probs = []
for i in range(n_classes):
    d_preds = cp.full(len(d_X_test), initial_predictions[i], dtype=cp.float32)
    for stump in all_class_stumps[i]:
        d_preds += lr * cp.where(d_X_test[:, stump['feat']] <= stump['thresh'], stump['left'], stump['right'])
    all_class_probs.append(1 / (1 + cp.exp(-d_preds)))

final_preds = cp.argmax(cp.stack(all_class_probs, axis=1), axis=1).get()
accuracy = np.mean(final_preds == y_test)

print(f"\nGradient Boosting (SVD-Embedded) Final Accuracy: {accuracy * 100:.2f}%")
print("------------------------------------------\n")