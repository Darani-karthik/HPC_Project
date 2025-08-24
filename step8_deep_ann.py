# step8_deep_ann.py (Ultra-Compact Hybrid Version)
# Added a comment 

import numpy as np
import scipy.sparse
import cupy as cp
import cupyx.scipy.sparse as cp_sparse

print("--- Running Step 8: Deeper ANN (Ultra-Compact Hybrid) ---")

# 1. Load Data & Params
X_sparse_cpu = scipy.sparse.load_npz('preprocessed_features.npz')
y_cpu = np.load('preprocessed_labels.npy')
n_samples, n_features = X_sparse_cpu.shape
n_classes = len(np.unique(y_cpu))
h1_size, h2_size = 128, 64
lr, epochs = 0.05, 1500

# 2. THE ONE CUSTOM KERNEL: Sparse Weight Update (X.T @ delta)
update_sparse_weights_kernel_code = r'''
extern "C" __global__
void update_sparse_weights(float* w1, float* b1, const float* delta_h1,
                           const float* X_data, const int* X_indices, const int* X_indptr,
                           float lr, int n_samples, int n_features, int h1_size) {
    int feature_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int h1_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (feature_idx < n_features && h1_idx < h1_size) {
        float grad_w1 = 0.0f;
        for (int i = 0; i < n_samples; ++i) {
            float feature_val = 0.0f;
            for (int j = X_indptr[i]; j < X_indptr[i + 1]; ++j) {
                if (X_indices[j] == feature_idx) {
                    feature_val = X_data[j];
                    break;
                }
            }
            grad_w1 += feature_val * delta_h1[i * h1_size + h1_idx];
        }
        w1[feature_idx * h1_size + h1_idx] -= lr * (grad_w1 / n_samples);

        if (feature_idx == 0) { // Update bias
            float grad_b1 = 0.0f;
            for (int i = 0; i < n_samples; ++i) {
                grad_b1 += delta_h1[i * h1_size + h1_idx];
            }
            b1[h1_idx] -= lr * (grad_b1 / n_samples);
        }
    }
}
'''
update_sparse_weights_kernel = cp.RawKernel(update_sparse_weights_kernel_code, 'update_sparse_weights')

# 3. Init Weights & Move data to GPU
w1 = (cp.random.randn(n_features, h1_size) * cp.sqrt(2./n_features)).astype(cp.float32)
b1 = cp.zeros(h1_size, dtype=cp.float32)
w2 = (cp.random.randn(h1_size, h2_size) * cp.sqrt(2./h1_size)).astype(cp.float32)
b2 = cp.zeros(h2_size, dtype=cp.float32)
w3 = (cp.random.randn(h2_size, n_classes) * np.sqrt(2./h2_size)).astype(cp.float32)
b3 = cp.zeros(n_classes, dtype=cp.float32)

d_X_sparse = cp_sparse.csr_matrix(X_sparse_cpu)
d_X_data = cp.asarray(X_sparse_cpu.data)
d_X_indices = cp.asarray(X_sparse_cpu.indices)
d_X_indptr = cp.asarray(X_sparse_cpu.indptr)
d_y_one_hot = cp.asarray(np.eye(n_classes)[y_cpu], dtype=cp.float32)

blocks_2d_w1 = ((h1_size + 15)//16, (n_features + 15)//16)
threads_2d = (16, 16)

# 4. Training Loop
print(f"Network: {n_features}->{h1_size}->{h2_size}->{n_classes}. Training...")
for epoch in range(epochs):
    # --- Forward Pass (High-level CuPy) ---
    h1_raw = d_X_sparse.dot(w1) + b1
    h1_act = cp.maximum(0, h1_raw) # ReLU
    h2_raw = h1_act.dot(w2) + b2
    h2_act = cp.maximum(0, h2_raw) # ReLU
    scores = h2_act.dot(w3) + b3
    exp_scores = cp.exp(scores - scores.max(axis=1, keepdims=True))
    probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
    
    # --- Backward Pass (High-level CuPy) ---
    delta_out = (probs - d_y_one_hot)
    delta_h2 = delta_out.dot(w3.T) * (h2_raw > 0)
    delta_h1 = delta_h2.dot(w2.T) * (h1_raw > 0)
    
    # --- Update Weights ---
    w3 -= lr * (h2_act.T.dot(delta_out) / n_samples)
    b3 -= lr * (cp.sum(delta_out, axis=0) / n_samples)
    w2 -= lr * (h1_act.T.dot(delta_h2) / n_samples)
    b2 -= lr * (cp.sum(delta_h2, axis=0) / n_samples)
    
    # Update sparse layer (w1, b1) with our CUSTOM KERNEL
    update_sparse_weights_kernel(blocks_2d_w1, threads_2d,
        (w1, b1, delta_h1, d_X_data, d_X_indices, d_X_indptr,
         lr, n_samples, n_features, h1_size))

    if (epoch + 1) % 100 == 0: print(f"Epoch {epoch+1}/{epochs} complete.")

# 5. Evaluation
print("Training finished.")
w1,b1,w2,b2,w3,b3 = w1.get(),b1.get(),w2.get(),b2.get(),w3.get(),b3.get()

h1 = np.maximum(0, X_sparse_cpu.dot(w1) + b1)
h2 = np.maximum(0, h1.dot(w2) + b2)
scores = h2.dot(w3) + b3
exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
preds = np.argmax(probs, axis=1)
accuracy = np.mean(preds == y_cpu)

print(f"\nDeeper ANN Final Accuracy: {accuracy * 100:.2f}%")
print("------------------------------------------\n")