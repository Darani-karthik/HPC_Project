
import numpy as np
import scipy.sparse
import cupy as cp
import cupyx.scipy.sparse as cp_sparse
from sklearn.metrics import classification_report, confusion_matrix

# Hyperparameters
HIDDEN_SIZE = 64
LEARNING_RATE = 0.15
EPOCHS = 2000

# Load data
X_sparse_cpu = scipy.sparse.load_npz('preprocessed_features.npz')
y_cpu = np.load('preprocessed_labels.npy')
n_samples, n_features = X_sparse_cpu.shape
n_classes = len(np.unique(y_cpu))
hidden_size, lr, epochs = HIDDEN_SIZE, LEARNING_RATE, EPOCHS

# CUDA kernel for sparse weight update
update_sparse_weights_kernel_code = r'''
extern "C" __global__
void update_sparse_weights(float* w1, float* b1, const float* delta_hidden,
                           const float* X_data, const int* X_indices, const int* X_indptr,
                           float lr, int n_samples, int n_features, int hidden_size) {
    int feature_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int hidden_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (feature_idx < n_features && hidden_idx < hidden_size) {
        float grad_w1 = 0.0f;
        for (int i = 0; i < n_samples; ++i) {
            float feature_val = 0.0f;
            for (int j = X_indptr[i]; j < X_indptr[i + 1]; ++j) {
                if (X_indices[j] == feature_idx) {
                    feature_val = X_data[j];
                    break;
                }
            }
            grad_w1 += feature_val * delta_hidden[i * hidden_size + hidden_idx];
        }
        w1[feature_idx * hidden_size + hidden_idx] -= lr * (grad_w1 / n_samples);

        if (feature_idx == 0) {
            float grad_b1 = 0.0f;
            for (int i = 0; i < n_samples; ++i) {
                grad_b1 += delta_hidden[i * hidden_size + hidden_idx];
            }
            b1[hidden_idx] -= lr * (grad_b1 / n_samples);
        }
    }
}
'''
update_sparse_weights_kernel = cp.RawKernel(update_sparse_weights_kernel_code, 'update_sparse_weights')

# Initialize weights and biases
w1 = (cp.random.randn(n_features, hidden_size) * cp.sqrt(2./n_features)).astype(cp.float32)
b1 = cp.zeros(hidden_size, dtype=cp.float32)
w2 = (cp.random.randn(hidden_size, n_classes) * cp.sqrt(2./hidden_size)).astype(cp.float32)
b2 = cp.zeros(n_classes, dtype=cp.float32)

# Move data to GPU
d_X_sparse = cp_sparse.csr_matrix(X_sparse_cpu)
d_X_data = cp.asarray(X_sparse_cpu.data)
d_X_indices = cp.asarray(X_sparse_cpu.indices)
d_X_indptr = cp.asarray(X_sparse_cpu.indptr)
d_y_one_hot = cp.asarray(np.eye(n_classes)[y_cpu], dtype=cp.float32)

# CUDA kernel launch parameters
blocks_2d_w1 = ((hidden_size + 15)//16, (n_features + 15)//16)
threads_2d = (16, 16)

# Training loop
print("Starting training...")
for epoch in range(epochs):
    # Forward pass
    d_hidden_raw = d_X_sparse.dot(w1) + b1
    d_hidden_activated = cp.maximum(0, d_hidden_raw) 
    d_scores = d_hidden_activated.dot(w2) + b2
    exp_scores = cp.exp(d_scores - d_scores.max(axis=1, keepdims=True))
    d_probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
    # Backward pass
    d_delta_output = (d_probs - d_y_one_hot)
    d_delta_hidden = d_delta_output.dot(w2.T) * (d_hidden_raw > 0)
    # Update output layer weights
    w2 -= lr * (d_hidden_activated.T.dot(d_delta_output) / n_samples)
    b2 -= lr * (cp.sum(d_delta_output, axis=0) / n_samples)
    # Update input layer weights using custom CUDA kernel
    update_sparse_weights_kernel(blocks_2d_w1, threads_2d,
        (w1, b1, d_delta_hidden, d_X_data, d_X_indices, d_X_indptr,
         lr, n_samples, n_features, hidden_size))

    # Print progress every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}/{epochs} complete.")

print("Training finished.")
# Move weights back to CPU
w1, b1, w2, b2 = w1.get(), b1.get(), w2.get(), b2.get()

# Forward pass on CPU for evaluation
h1 = np.maximum(0, X_sparse_cpu.dot(w1) + b1)
scores = h1.dot(w2) + b2
exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
preds = np.argmax(probs, axis=1)
accuracy = np.mean(preds == y_cpu)

# Print evaluation metrics
print(f"\nANN Final Accuracy: {accuracy * 100:.2f}%")
print("Confusion Matrix:")
print(confusion_matrix(y_cpu, preds))
print("\nClassification Report:")
print(classification_report(y_cpu, preds, digits=4))