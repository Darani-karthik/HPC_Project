import numpy as np
import scipy.sparse
import cupy as cp
import cupyx.scipy.sparse as cp_sparse

# --- Load / prepare ---
X_sparse_cpu = scipy.sparse.load_npz('preprocessed_features.npz').tocsr()
y_cpu = np.load('preprocessed_labels.npy').astype(np.int32)

n_samples, n_features = X_sparse_cpu.shape
n_classes = int(len(np.unique(y_cpu)))
y_one_hot_cpu = np.eye(n_classes, dtype=np.float32)[y_cpu]

# Move CSR arrays to GPU 
d_X_data = cp.asarray(X_sparse_cpu.data.astype(np.float32))
d_X_indices = cp.asarray(X_sparse_cpu.indices.astype(np.int32))
d_X_indptr = cp.asarray(X_sparse_cpu.indptr.astype(np.int32))

d_X_sparse = cp_sparse.csr_matrix((d_X_data, d_X_indices, d_X_indptr), shape=(n_samples, n_features))
d_y_one_hot = cp.asarray(y_one_hot_cpu, dtype=cp.float32)

# weights on GPU
d_weights = cp.zeros((n_features, n_classes), dtype=cp.float32)

# hyperparams
lr = np.float32(0.2)
epochs = 1500

# --- Improved CUDA kernel ---
update_weights_kernel_code = r'''
extern "C" _global_
void update_weights_kernel(float* _restrict_ weights,
                           const float* _restrict_ X_data,
                           const int* _restrict_ X_indices,
                           const int* _restrict_ X_indptr,
                           const float* _restrict_ error,
                           float lr,
                           int n_samples,
                           int n_features,
                           int n_classes) {

    int sample_idx = blockIdx.x;
    if (sample_idx >= n_samples) return;

    int tid = threadIdx.x;

    int start = X_indptr[sample_idx];
    int end = X_indptr[sample_idx + 1];

    const float* err_row = &error[sample_idx * n_classes];

    for (int idx = start; idx < end; ++idx) {
        int feat = X_indices[idx];
        float x = X_data[idx];

        for (int class_idx = tid; class_idx < n_classes; class_idx += blockDim.x) {
            float g = x * err_row[class_idx];
            float delta = - (lr / (float)n_samples) * g;
            atomicAdd(&weights[feat * n_classes + class_idx], delta);
        }
    }
}
'''
update_weights_kernel = cp.RawKernel(update_weights_kernel_code, 'update_weights_kernel')

# --- Kernel launch config ---
threads_per_block = int(min(1024, max(32, n_classes)))
blocks_per_grid = (n_samples,)

# Preallocate intermediate arrays
d_scores = cp.empty((n_samples, n_classes), dtype=cp.float32)
exp_scores = cp.empty_like(d_scores)
d_probabilities = cp.empty_like(d_scores)
d_error = cp.empty_like(d_scores)

print("Starting training (custom kernel)...")
for epoch in range(epochs):
    # forward
    d_scores = d_X_sparse.dot(d_weights)               # (n_samples, n_classes)
    
    # stable softmax
    row_max = d_scores.max(axis=1, keepdims=True)
    exp_scores = cp.exp(d_scores - row_max)
    d_probabilities = exp_scores / exp_scores.sum(axis=1, keepdims=True)

    # error (prob - label)
    d_error = d_probabilities - d_y_one_hot

    # compute delta updates
    d_weights_delta = cp.zeros_like(d_weights)
    update_weights_kernel((n_samples,), (threads_per_block,),
                          (d_weights_delta, d_X_data, d_X_indices, d_X_indptr,
                           d_error, lr, n_samples, n_features, n_classes))

    # apply accumulated delta to weights
    d_weights += d_weights_delta

    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}/{epochs} complete.")

# --- Final weights to CPU ---
final_weights = d_weights.get()
print("Training finished.")

# --- Evaluation  ---
print("Evaluating on CPU...")

scores = X_sparse_cpu.dot(final_weights)  # shape (n_samples, n_classes)
exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
probabilities = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
predictions = np.argmax(probabilities, axis=1)

accuracy = np.mean(predictions == y_cpu)
print(f"\nSoftmax Regression (Custom CUDA Kernel) Final Accuracy: {accuracy * 100:.2f}%")

print("------------------------------------------\n")
