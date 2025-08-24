# step3_svm.py (Compact Hybrid CUDA & CuPy Version)

import numpy as np
import scipy.sparse
import cupy as cp
import cupyx.scipy.sparse as cp_sparse

print("--- Running Step 3: SVM (Hybrid CUDA & CuPy) ---")

# 1. Load Data & Set Params
X_sparse_cpu = scipy.sparse.load_npz('preprocessed_features.npz')
y_cpu = np.load('preprocessed_labels.npy')
n_samples, n_features = X_sparse_cpu.shape
n_classes = len(np.unique(y_cpu))
epochs, lr, C = 500, 0.01, 1.0

# 2. The Custom CUDA Kernel for the SVM Gradient Update
svm_update_kernel_code = r'''
extern "C" __global__
void svm_update_kernel(float* weights, const float* X_data, const int* X_indices, const int* X_indptr,
                       const float* y, const float* scores, float lr, float C,
                       int n_samples, int n_features) {
    int feature_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (feature_idx < n_features) {
        float grad = weights[feature_idx]; // Regularization
        for (int sample_idx = 0; sample_idx < n_samples; ++sample_idx) {
            if (y[sample_idx] * scores[sample_idx] < 1.0f) { // Hinge loss condition
                float feature_val = 0.0f;
                for (int i = X_indptr[sample_idx]; i < X_indptr[sample_idx + 1]; ++i) {
                    if (X_indices[i] == feature_idx) {
                        feature_val = X_data[i];
                        break; 
                    }
                }
                grad -= C * y[sample_idx] * feature_val;
            }
        }
        weights[feature_idx] -= lr * (grad / n_samples);
    }
}
'''
svm_update_kernel = cp.RawKernel(svm_update_kernel_code, 'svm_update_kernel')

# 3. Move data to GPU & Define Kernel Config
d_X_data = cp.asarray(X_sparse_cpu.data)
d_X_indices = cp.asarray(X_sparse_cpu.indices)
d_X_indptr = cp.asarray(X_sparse_cpu.indptr)
d_X_sparse = cp_sparse.csr_matrix(X_sparse_cpu, dtype=cp.float32)

threadsPB = (256,)
blocksPG_features = ((n_features + 255) // 256,)

# 4. Main Training Loop (One-vs-Rest)
all_weights = np.zeros((n_features, n_classes), dtype=np.float32)
for i in range(n_classes):
    print(f"\nTraining classifier for class {i}...")
    
    d_y_binary = cp.where(cp.asarray(y_cpu) == i, 1, -1).astype(cp.float32)
    d_weights_class = cp.zeros(n_features, dtype=cp.float32)

    for epoch in range(epochs):
        # Step A: Forward pass -> High-Level CuPy
        d_scores = d_X_sparse.dot(d_weights_class)
        
        # Step B: Update weights -> Custom CUDA Kernel
        svm_update_kernel(blocksPG_features, threadsPB,
            (d_weights_class, d_X_data, d_X_indices, d_X_indptr, d_y_binary, 
             d_scores, lr, C, n_samples, n_features)
        )

    all_weights[:, i] = d_weights_class.get()

# 5. Evaluation
print("\nTraining complete. Evaluating...")
final_scores = X_sparse_cpu.dot(all_weights)
predictions = np.argmax(final_scores, axis=1)
accuracy = np.mean(predictions == y_cpu)

print(f"\nSVM (One-vs-Rest) Final Accuracy: {accuracy * 100:.2f}%")
print("------------------------------------------\n")