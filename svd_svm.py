import numpy as np
import scipy.sparse
import cupy as cp
import cupyx.scipy.sparse as cp_sparse
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

print("--- Running Step 3: SVM (Hybrid CUDA & CuPy) on fin_data_1.csv ---")

# ============================================================== 
# 1. Load Data & Set Params (Integrated Preprocessing)
# ----------------------------------------------------
print("Loading fin_data_1.csv and performing TF-IDF Vectorization...")

# Load data
df = pd.read_csv('fin_data_1.csv')

# Map Sentiment to numerical labels (+1, 0, -1 are internally used as 0, 1, 2)
sentiment_map = {'negative': 0, 'neutral': 1, 'positive': 2}
df['Sentiment_Numeric'] = df['Sentiment'].map(sentiment_map)
y_cpu = df['Sentiment_Numeric'].values.astype(np.int32)

# TF-IDF Vectorization: Creates the sparse feature matrix X
# Limiting features to max_features=5000 to keep the matrix manageable for GPU memory.
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_sparse_cpu = vectorizer.fit_transform(df['Sentence'])

# Define the core hyperparameters and extracted data properties
n_samples, n_features = X_sparse_cpu.shape
n_classes = len(np.unique(y_cpu))
epochs, lr, C = 500, 0.01, 1.0
print(f"Data loaded and vectorized. Features shape: {X_sparse_cpu.shape}, Classes: {n_classes}")

# ============================================================== 
# 2. The Custom CUDA Kernel for the SVM Gradient Update
# ----------------------------------------------------
# Keep your original kernel logic, but ensure types are used consistently.
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
        weights[feature_idx] -= lr * (grad / (float)n_samples);
    }
}
'''
# Compile the CUDA code into a callable kernel object using CuPy.
svm_update_kernel = cp.RawKernel(svm_update_kernel_code, 'svm_update_kernel')

# ============================================================== 
# 3. Move data to GPU & Define Kernel Config
# -------------------------------------------
# Convert SciPy CSR arrays to appropriate dtypes and push to GPU
d_X_data = cp.asarray(X_sparse_cpu.data.astype(np.float32))
d_X_indices = cp.asarray(X_sparse_cpu.indices.astype(np.int32))
d_X_indptr = cp.asarray(X_sparse_cpu.indptr.astype(np.int32))

# Build a CuPy CSR matrix from the components
d_X_sparse = cp_sparse.csr_matrix((d_X_data, d_X_indices, d_X_indptr), shape=(n_samples, n_features))

# Prepare GPU versions of constants
# Note: we'll create per-class label arrays below inside the loop
threads_per_block = 256
num_blocks = (n_features + threads_per_block - 1) // threads_per_block
grid = (num_blocks,)   # grid must be a tuple for RawKernel
block = (threads_per_block,)

# ============================================================== 
# 4. Main Training Loop (One-vs-Rest)
# -----------------------------------
all_weights = np.zeros((n_features, n_classes), dtype=np.float32)

# Pre-convert scalar hyperparams to correct numpy types for passing
lr_f32 = np.float32(lr)
C_f32 = np.float32(C)
n_samples_i32 = np.int32(n_samples)
n_features_i32 = np.int32(n_features)

# Move a CPU array of indices for argmax evaluation later if needed
for i in range(n_classes):
    print(f"\nTraining classifier for class {i}...")

    # For the current class `i`, create a binary label vector on the GPU (+1 for class `i`, -1 for all others).
    d_y_binary = cp.where(cp.asarray(y_cpu) == i, 1, -1).astype(cp.float32)

    # Initialize weights for this class on the GPU
    d_weights_class = cp.zeros(n_features, dtype=cp.float32)

    # Main training loop for the individual classifier.
    for epoch in range(epochs):
        # Step A: Forward pass (High-Level CuPy) - Calculate prediction scores.
        # d_scores shape: (n_samples,)
        d_scores = d_X_sparse.dot(d_weights_class).astype(cp.float32)

        # Step B: Update weights (Custom CUDA Kernel) - Perform gradient calculation and weight updates.
        # Prepare args tuple (note the order must match kernel signature)
        args = (
            d_weights_class,      # float* weights
            d_X_data,             # const float* X_data
            d_X_indices,          # const int* X_indices
            d_X_indptr,           # const int* X_indptr
            d_y_binary,           # const float* y
            d_scores,             # const float* scores
            lr_f32,               # float lr
            C_f32,                # float C
            n_samples_i32,        # int n_samples
            n_features_i32        # int n_features
        )

        # Kernel launch: grid and block are tuples.
        svm_update_kernel(grid, block, args)

    # After training, retrieve the finalized weights for this classifier.
    all_weights[:, i] = d_weights_class.get()

# ============================================================== 
# 5. Evaluation
# --------------------------------------------------------------
print("\nTraining complete. Evaluating...")
# final_scores = X_sparse_cpu.dot(all_weights)  # SciPy sparse dot with dense matrix
# SciPy dot returns float64 by default — convert to float32 for memory
final_scores = X_sparse_cpu.dot(all_weights).astype(np.float32)
predictions = np.argmax(final_scores, axis=1)
accuracy = np.mean(predictions == y_cpu)

print(f"\nSVM (One-vs-Rest) Final Accuracy: {accuracy * 100:.2f}%")
print("------------------------------------------\n")
