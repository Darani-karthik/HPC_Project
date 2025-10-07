# step3_svm.py (Hybrid CUDA & CuPy Version - Fixed and Tuned)
import numpy as np
import scipy.sparse
import cupy as cp
import cupyx.scipy.sparse as cp_sparse
from sklearn.preprocessing import normalize

print("--- Running Step 3: SVM (Hybrid CUDA & CuPy) ---")

# 1. Load Data & Set Params
X_sparse_cpu = scipy.sparse.load_npz('preprocessed_features.npz').tocsr()  # ✅ ensure CSR format
y_cpu = np.load('preprocessed_labels.npy')

# Normalize feature matrix
X_sparse_cpu = normalize(X_sparse_cpu, norm='l2', axis=1)

# Add bias term as extra column
bias = scipy.sparse.csr_matrix(np.ones((X_sparse_cpu.shape[0], 1), dtype=np.float32))
X_sparse_cpu = scipy.sparse.hstack([X_sparse_cpu, bias], format='csr')

n_samples, n_features = X_sparse_cpu.shape
n_classes = len(np.unique(y_cpu))
epochs, lr, C = 800, 0.001, 10.0  # tuned hyperparameters

print(f"Data loaded: {n_samples} samples, {n_features} features (+bias), {n_classes} classes")

# 2. Custom CUDA Kernel for Gradient Update
svm_update_kernel_code = r'''
extern "C" __global__
void svm_update_kernel(float* weights, const float* X_data, const int* X_indices, const int* X_indptr,
                       const float* y, const float* scores, float lr, float C,
                       int n_samples, int n_features) {
    int feature_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (feature_idx < n_features) {
        float grad = weights[feature_idx]; // Regularization term
        for (int sample_idx = 0; sample_idx < n_samples; ++sample_idx) {
            if (y[sample_idx] * scores[sample_idx] < 1.0f) { // hinge loss condition
                for (int i = X_indptr[sample_idx]; i < X_indptr[sample_idx + 1]; ++i) {
                    if (X_indices[i] == feature_idx) {
                        grad -= C * y[sample_idx] * X_data[i];
                        break;
                    }
                }
            }
        }
        weights[feature_idx] -= lr * (grad / (float)n_samples);
    }
}
'''
svm_update_kernel = cp.RawKernel(svm_update_kernel_code, 'svm_update_kernel')

# 3. Move data to GPU & Define Kernel Config
d_X_data = cp.asarray(X_sparse_cpu.data.astype(np.float32))
d_X_indices = cp.asarray(X_sparse_cpu.indices.astype(np.int32))
d_X_indptr = cp.asarray(X_sparse_cpu.indptr.astype(np.int32))
d_X_sparse = cp_sparse.csr_matrix((d_X_data, d_X_indices, d_X_indptr),
                                  shape=(n_samples, n_features), dtype=cp.float32)

threadsPB = (256,)
blocksPG_features = ((n_features + threadsPB[0] - 1) // threadsPB[0],)

# 4. Main Training Loop (One-vs-Rest)
all_weights = np.zeros((n_features, n_classes), dtype=np.float32)

for i in range(n_classes):
    print(f"\nTraining classifier for class {i}...")
    
    d_y_binary = cp.where(cp.asarray(y_cpu) == i, 1, -1).astype(cp.float32)
    d_weights_class = cp.zeros(n_features, dtype=cp.float32)

    for epoch in range(1, epochs + 1):
        # Step A: Forward pass
        d_scores = d_X_sparse.dot(d_weights_class)

        # Step B: Kernel weight update
        svm_update_kernel(
            blocksPG_features, threadsPB,
            (
                d_weights_class, d_X_data, d_X_indices, d_X_indptr,
                d_y_binary, d_scores, np.float32(lr), np.float32(C),
                np.int32(n_samples), np.int32(n_features)
            )
        )

        # Optional logging every 100 epochs
        if epoch % 100 == 0 or epoch == 1:
            preds = cp.where(d_scores >= 0, 1, -1)
            acc = float(cp.mean((preds == d_y_binary).astype(cp.float32)).get())
            print(f"  Epoch {epoch}/{epochs} - Training Acc: {acc*100:.2f}%")

    all_weights[:, i] = d_weights_class.get()

# 5. Evaluation
print("\nTraining complete. Evaluating...")
final_scores = X_sparse_cpu.dot(all_weights)
predictions = np.argmax(final_scores, axis=1)
accuracy = np.mean(predictions == y_cpu)

print(f"\nSVM (One-vs-Rest) Final Accuracy: {accuracy * 100:.2f}%")
print("------------------------------------------\n")
