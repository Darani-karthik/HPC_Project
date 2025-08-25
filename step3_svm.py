import numpy as np
import scipy.sparse
import cupy as cp
import cupyx.scipy.sparse as cp_sparse

print("--- Running Step 3: SVM (Hybrid CUDA & CuPy) ---")

# 1. Load Data & Set Params
# -------------------------
# Load the sparse feature matrix (X) and labels (y) from disk into CPU memory.
# We also define the core hyperparameters for our SVM model:
# - epochs: How many times to iterate over the entire dataset.
# - lr: The learning rate, controlling the size of our optimization steps.
# - C: The regularization parameter, which balances classification accuracy on the
#      training set against the model's ability to generalize to new data.
X_sparse_cpu = scipy.sparse.load_npz('preprocessed_features.npz')
y_cpu = np.load('preprocessed_labels.npy')
n_samples, n_features = X_sparse_cpu.shape
n_classes = len(np.unique(y_cpu))
epochs, lr, C = 500, 0.01, 1.0

# 2. The Custom CUDA Kernel for the SVM Gradient Update
# ----------------------------------------------------
# This is a custom CUDA C++ kernel written as a Python raw string. It's designed for
# massive parallelism on the GPU. Each thread on the GPU will be responsible for
# calculating the gradient and updating the weight of a single feature.
# The core logic inside is the SVM's hinge loss, which only considers samples that
# are misclassified or within the classification margin (y * score < 1), making the
# update efficient. This low-level control is ideal for the most performance-critical
# part of the training loop.
svm_update_kernel_code = r'''
extern "C" __global__
void svm_update_kernel(float* weights, const float* X_data, const int* X_indices, const int* X_indptr,
                       const float* y, const float* scores, float lr, float C,
                       int n_samples, int n_features) {
    int feature_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (feature_idx < n_features) {
        float grad = weights[feature_idx]; // Regularization term
        // Hinge loss gradient calculation
        for (int sample_idx = 0; sample_idx < n_samples; ++sample_idx) {
            if (y[sample_idx] * scores[sample_idx] < 1.0f) { // Condition for support vectors
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
        // Apply the weight update
        weights[feature_idx] -= lr * (grad / n_samples);
    }
}
'''
# Compile the CUDA code into a callable kernel object using CuPy.
svm_update_kernel = cp.RawKernel(svm_update_kernel_code, 'svm_update_kernel')

# 3. Move data to GPU & Define Kernel Config
# -------------------------------------------
# Transfer the underlying components of the SciPy sparse matrix to the GPU.
# We then reconstruct the sparse matrix in GPU memory using CuPy's sparse library
# for use in high-level operations like matrix multiplication.
d_X_data = cp.asarray(X_sparse_cpu.data)
d_X_indices = cp.asarray(X_sparse_cpu.indices)
d_X_indptr = cp.asarray(X_sparse_cpu.indptr)
d_X_sparse = cp_sparse.csr_matrix(X_sparse_cpu, dtype=cp.float32)

# Configure the launch parameters for our custom CUDA kernel. This defines
# the number of threads per block and the total number of blocks in the grid,
# ensuring we have enough threads to process all features in parallel.
threadsPB = (256,)
blocksPG_features = ((n_features + 255) // 256,)

# 4. Main Training Loop (One-vs-Rest)
# -----------------------------------
# To handle a multi-class problem with a binary classifier like SVM, we use the
# "One-vs-Rest" (OvR) strategy. We will train a separate classifier for each class,
# where each one is trained to distinguish its class from all other classes combined.
all_weights = np.zeros((n_features, n_classes), dtype=np.float32)
for i in range(n_classes):
    print(f"\nTraining classifier for class {i}...")
    
    # For the current class `i`, create a binary label vector on the GPU (+1 for class `i`, -1 for all others).
    d_y_binary = cp.where(cp.asarray(y_cpu) == i, 1, -1).astype(cp.float32)
    d_weights_class = cp.zeros(n_features, dtype=cp.float32)

    # Main training loop for the individual classifier.
    for epoch in range(epochs):
        # Step A: Forward pass (High-Level CuPy).
        # Calculate the current prediction scores for all samples. This is a fast,
        # optimized matrix-vector multiplication handled efficiently by CuPy's sparse library.
        d_scores = d_X_sparse.dot(d_weights_class)
        
        # Step B: Update weights (Custom CUDA Kernel).
        # Launch our custom kernel to perform the complex gradient calculation and
        # weight updates in parallel across all features on the GPU.
        svm_update_kernel(blocksPG_features, threadsPB,
            (d_weights_class, d_X_data, d_X_indices, d_X_indptr, d_y_binary, 
             d_scores, lr, C, n_samples, n_features)
        )

    # After training, retrieve the finalized weights for this classifier from the GPU back to the CPU.
    all_weights[:, i] = d_weights_class.get()

# 5. Evaluation
# -------------
# With all classifiers trained, we make predictions on the full dataset.
# The final prediction for each sample is the class whose corresponding classifier
# produced the highest score. We then compare these predictions to the true labels
# to calculate the final model accuracy.
print("\nTraining complete. Evaluating...")
final_scores = X_sparse_cpu.dot(all_weights)
predictions = np.argmax(final_scores, axis=1)
accuracy = np.mean(predictions == y_cpu)

print(f"\nSVM (One-vs-Rest) Final Accuracy: {accuracy * 100:.2f}%")
print("------------------------------------------\n")
