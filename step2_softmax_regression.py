# step2_softmax_regression.py 

import numpy as np
import scipy.sparse
import cupy as cp
import cupyx.scipy.sparse as cp_sparse


print("--- Running Step 2: Softmax Regression (Hybrid CUDA & CuPy) ---")
print("NOTE: Using a custom kernel for the gradient calculation and high-level CuPy for the rest.")

# 1. Load Data
X_sparse_cpu = scipy.sparse.load_npz('preprocessed_features.npz').tocsr()
y_cpu = np.load('preprocessed_labels.npy') # Variable is named y_cpu

# 2. Prepare Data
n_samples, n_features = X_sparse_cpu.shape
n_classes = len(np.unique(y_cpu))
y_one_hot_cpu = np.eye(n_classes)[y_cpu].astype(np.float32)

# 3. Set Hyperparameters
lr = 0.2
epochs = 1500

# 4. The Custom CUDA Kernel

update_weights_kernel_code = r'''
extern "C" __global__
void update_weights_kernel(float* weights,
                           const float* X_data, const int* X_indices, const int* X_indptr, 
                           const float* error, float lr,
                           int n_samples, int n_features, int n_classes) {

    int feature_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int class_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (feature_idx < n_features && class_idx < n_classes) {
        float grad = 0.0f;

        for (int sample_idx = 0; sample_idx < n_samples; ++sample_idx) {
            float feature_val = 0.0f;
            for (int i = X_indptr[sample_idx]; i < X_indptr[sample_idx + 1]; ++i) {
                if (X_indices[i] == feature_idx) {
                    feature_val = X_data[i];
                    break; 
                }
            }
            grad += feature_val * error[sample_idx * n_classes + class_idx];
        }
        
        weights[feature_idx * n_classes + class_idx] -= lr * (grad / n_samples);
    }
}
'''
update_weights_kernel = cp.RawKernel(update_weights_kernel_code, 'update_weights_kernel')

# 5. Move data to GPU & Define Kernel Config
d_X_data = cp.asarray(X_sparse_cpu.data)
d_X_indices = cp.asarray(X_sparse_cpu.indices)
d_X_indptr = cp.asarray(X_sparse_cpu.indptr)
d_X_sparse = cp_sparse.csr_matrix(X_sparse_cpu, dtype=cp.float32)

d_y_one_hot = cp.asarray(y_one_hot_cpu)
d_weights = cp.zeros((n_features, n_classes), dtype=cp.float32)

threads_per_block_2d = (16, 16)
blocks_per_grid_2d = ((n_features + 15) // 16, (n_classes + 15) // 16)

# 6. Training Loop 
print("Starting training...")
for epoch in range(epochs):
    d_scores = d_X_sparse.dot(d_weights)

    exp_scores = cp.exp(d_scores - d_scores.max(axis=1, keepdims=True))
    d_probabilities = exp_scores / exp_scores.sum(axis=1, keepdims=True)
    d_error = d_probabilities - d_y_one_hot

    update_weights_kernel(blocks_per_grid_2d, threads_per_block_2d,
                          (d_weights, d_X_data, d_X_indices, d_X_indptr, d_error, lr,
                           n_samples, n_features, n_classes))

    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}/{epochs} complete.")

# 7. Get final weights from GPU
final_weights = d_weights.get()
print("Training finished.")

# 8. Evaluation (Using the original NumPy/SciPy)
scores = X_sparse_cpu.dot(final_weights)
exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
probabilities = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
predictions = np.argmax(probabilities, axis=1)
accuracy = np.mean(predictions == y_cpu)

print(f"\nSoftmax Regression Final Accuracy: {accuracy * 100:.2f}%")

print("------------------------------------------\n")
