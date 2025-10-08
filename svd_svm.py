import numpy as np
import scipy.sparse
import cupy as cp
import cupyx.scipy.sparse as cp_sparse
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD # Import TruncatedSVD

print("--- Running Step 3: SVM (Hybrid CUDA & CuPy) with SVD Embedding ---")

# ==============================================================
# 1. Load Data & Preprocessing
# ----------------------------------------------------
print("Loading fin_data_1.csv and performing initial processing...")

# Load data
df = pd.read_csv('fin_data_1.csv')

# Map Sentiment to numerical labels (+1, 0, -1 are internally used as 0, 1, 2)
sentiment_map = {'negative': 0, 'neutral': 1, 'positive': 2}
df['Sentiment_Numeric'] = df['Sentiment'].map(sentiment_map)
y_cpu = df['Sentiment_Numeric'].values.astype(np.int32)

# ==============================================================
# 2. TF-IDF Vectorization & SVD Embedding
# ----------------------------------------------------
print("Performing TF-IDF Vectorization followed by SVD Embedding...")

# Step A: TF-IDF Vectorization to create a sparse feature matrix
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_sparse_cpu = vectorizer.fit_transform(df['Sentence'])
print(f"Shape after TF-IDF: {X_sparse_cpu.shape}")

# Step B: SVD to create a dense embedding from the sparse matrix
# We reduce the 5000 TF-IDF features to 300 dense semantic features.
n_components = 300
svd = TruncatedSVD(n_components=n_components, random_state=42)
X_dense_cpu = svd.fit_transform(X_sparse_cpu).astype(np.float32)

# Update data properties based on the new dense matrix
n_samples, n_features = X_dense_cpu.shape
n_classes = len(np.unique(y_cpu))
epochs, lr, C = 500, 0.01, 1.0
print(f"Shape after SVD Embedding: {X_dense_cpu.shape}, Classes: {n_classes}")

# ==============================================================
# 3. Custom CUDA Kernel for DENSE SVM Gradient Update
# ----------------------------------------------------
# This kernel is now simpler because it operates on a dense matrix.
svm_update_kernel_code_dense = r'''
extern "C" __global__
void svm_update_kernel_dense(float* weights, const float* X_dense,
                             const float* y, const float* scores, float lr, float C,
                             int n_samples, int n_features) {
    int feature_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (feature_idx < n_features) {
        float grad = weights[feature_idx]; // Regularization term

        // Hinge loss gradient calculation across all samples
        for (int sample_idx = 0; sample_idx < n_samples; ++sample_idx) {
            // Check if the sample is a support vector
            if (y[sample_idx] * scores[sample_idx] < 1.0f) {
                // For a dense matrix, access is direct: X[row * num_cols + col]
                grad -= C * y[sample_idx] * X_dense[sample_idx * n_features + feature_idx];
            }
        }
        // Apply the weight update
        weights[feature_idx] -= lr * (grad / (float)n_samples);
    }
}
'''
# Compile the new CUDA code for dense matrices.
svm_update_kernel = cp.RawKernel(svm_update_kernel_code_dense, 'svm_update_kernel_dense')

# ==============================================================
# 4. Move data to GPU & Define Kernel Config
# -------------------------------------------
# Move the dense matrix to the GPU
d_X = cp.asarray(X_dense_cpu)

# Kernel launch configuration (remains the same, but n_features is now 300)
threads_per_block = 256
num_blocks = (n_features + threads_per_block - 1) // threads_per_block
grid = (num_blocks,)
block = (threads_per_block,)

# ==============================================================
# 5. Main Training Loop (One-vs-Rest)
# -----------------------------------
all_weights = np.zeros((n_features, n_classes), dtype=np.float32)

# Pre-convert scalar hyperparams to correct types
lr_f32 = np.float32(lr)
C_f32 = np.float32(C)
n_samples_i32 = np.int32(n_samples)
n_features_i32 = np.int32(n_features)

for i in range(n_classes):
    print(f"\nTraining classifier for class {i}...")

    # Create a binary label vector on the GPU (+1 for class `i`, -1 for others).
    d_y_binary = cp.where(cp.asarray(y_cpu) == i, 1, -1).astype(cp.float32)

    # Initialize weights for this class on the GPU
    d_weights_class = cp.zeros(n_features, dtype=cp.float32)

    # Main training loop for the individual classifier.
    for epoch in range(epochs):
        # Step A: Forward pass - Calculate prediction scores using dense matrix multiplication.
        d_scores = cp.dot(d_X, d_weights_class)

        # Step B: Update weights - Call the custom kernel for dense matrices.
        args = (
            d_weights_class,      # float* weights
            d_X,                  # const float* X_dense
            d_y_binary,           # const float* y
            d_scores,             # const float* scores
            lr_f32,               # float lr
            C_f32,                # float C
            n_samples_i32,        # int n_samples
            n_features_i32        # int n_features
        )
        svm_update_kernel(grid, block, args)

    # Retrieve the finalized weights for this classifier.
    all_weights[:, i] = d_weights_class.get()

# ==============================================================
# 6. Evaluation
# --------------------------------------------------------------
print("\nTraining complete. Evaluating...")
# Use the dense CPU matrix for final evaluation
final_scores = X_dense_cpu.dot(all_weights)
predictions = np.argmax(final_scores, axis=1)
accuracy = np.mean(predictions == y_cpu)

print(f"\nSVM (One-vs-Rest with SVD) Final Accuracy: {accuracy * 100:.2f}%")
print("------------------------------------------\n")
