import numpy as np
import scipy.sparse
from scipy.sparse.linalg import svds
import cupy as cp
import cupyx.scipy.sparse as cp_sparse

# --- Load / prepare ---
X_sparse_cpu = scipy.sparse.load_npz('preprocessed_features.npz').tocsr()
y_cpu = np.load('preprocessed_labels.npy').astype(np.int32)

n_samples, n_features = X_sparse_cpu.shape
n_classes = int(len(np.unique(y_cpu)))
y_one_hot_cpu = np.eye(n_classes, dtype=np.float32)[y_cpu]

# --- SVD-based embedding generation ---
print("Computing SVD embeddings...")
embedding_dim = min(512, n_features - 1)  # Adjust as needed

# Compute truncated SVD
U, S, Vt = svds(X_sparse_cpu.astype(np.float32), k=embedding_dim)

# Generate embeddings: X_embedded = X @ V (where V = Vt.T)
# V represents the embedding matrix (n_features x embedding_dim)
embedding_matrix = Vt.T  # (n_features, embedding_dim)

# Transform data to embedded space
X_embedded_cpu = X_sparse_cpu.dot(embedding_matrix).astype(np.float32)
print(f"Original features: {n_features}, Embedded dimension: {embedding_dim}")
print(f"Explained variance ratio: {(S**2).sum() / np.sum(X_sparse_cpu.data**2):.4f}")

# Move embedded data to GPU (now dense)
d_X_embedded = cp.asarray(X_embedded_cpu, dtype=cp.float32)
d_y_one_hot = cp.asarray(y_one_hot_cpu, dtype=cp.float32)

# Weights in embedded space
d_weights = cp.zeros((embedding_dim, n_classes), dtype=cp.float32)

# hyperparams
lr = np.float32(0.2)
epochs = 1500

# --- Tiled CUDA kernel for dense matrix ---
update_weights_dense_tiled_kernel_code = r'''
extern "C" __global__
void update_weights_dense_tiled_kernel(float* __restrict__ weights,
                                        const float* __restrict__ X_embedded,
                                        const float* __restrict__ error,
                                        float lr,
                                        int n_samples,
                                        int embedding_dim,
                                        int n_classes,
                                        int tile_size) {

    int sample_idx = blockIdx.x;
    if (sample_idx >= n_samples) return;

    int tid = threadIdx.x;

    // Shared memory for tiling
    extern __shared__ float shared_mem[];
    float* shared_error = shared_mem;
    float* shared_x = &shared_mem[tile_size];

    const float* x_row = &X_embedded[sample_idx * embedding_dim];
    const float* err_row = &error[sample_idx * n_classes];
    
    float scale = - (lr / (float)n_samples);

    // Process classes in tiles
    for (int class_tile = 0; class_tile < n_classes; class_tile += tile_size) {
        int class_tile_end = min(class_tile + tile_size, n_classes);
        int class_tile_size = class_tile_end - class_tile;
        
        // Load error tile into shared memory
        for (int i = tid; i < class_tile_size; i += blockDim.x) {
            shared_error[i] = err_row[class_tile + i];
        }
        __syncthreads();

        // Process embedding dimensions in tiles
        for (int feat_tile = 0; feat_tile < embedding_dim; feat_tile += tile_size) {
            int feat_tile_end = min(feat_tile + tile_size, embedding_dim);
            int feat_tile_size = feat_tile_end - feat_tile;
            
            // Load feature tile into shared memory
            for (int i = tid; i < feat_tile_size; i += blockDim.x) {
                shared_x[i] = x_row[feat_tile + i];
            }
            __syncthreads();

            // Compute gradients for this tile
            for (int f_idx = 0; f_idx < feat_tile_size; ++f_idx) {
                int feat = feat_tile + f_idx;
                float x = shared_x[f_idx];

                for (int c_idx = tid; c_idx < class_tile_size; c_idx += blockDim.x) {
                    int class_idx = class_tile + c_idx;
                    float g = x * shared_error[c_idx];
                    float delta = scale * g;
                    atomicAdd(&weights[feat * n_classes + class_idx], delta);
                }
            }
            __syncthreads();
        }
    }
}
'''

update_weights_kernel = cp.RawKernel(update_weights_dense_tiled_kernel_code, 
                                     'update_weights_dense_tiled_kernel')

# Kernel configuration
tile_size = 128
threads_per_block = 256
shared_mem_bytes = tile_size * 2 * 4  # error + features
blocks_per_grid = (n_samples,)

print(f"\nKernel: Dense SVD-Embedded Tiled")
print(f"Tile size: {tile_size}")
print(f"Threads per block: {threads_per_block}")
print(f"Shared memory per block: {shared_mem_bytes} bytes")

# Preallocate intermediate arrays
d_scores = cp.empty((n_samples, n_classes), dtype=cp.float32)
exp_scores = cp.empty_like(d_scores)
d_probabilities = cp.empty_like(d_scores)
d_error = cp.empty_like(d_scores)

print("\nStarting training with SVD embeddings...")
for epoch in range(epochs):
    # Forward pass: X_embedded @ weights
    d_scores = cp.dot(d_X_embedded, d_weights)
    
    # Stable softmax
    row_max = d_scores.max(axis=1, keepdims=True)
    exp_scores = cp.exp(d_scores - row_max)
    d_probabilities = exp_scores / exp_scores.sum(axis=1, keepdims=True)

    # Error (prob - label)
    d_error = d_probabilities - d_y_one_hot

    # Compute delta updates with tiled kernel
    d_weights_delta = cp.zeros_like(d_weights)
    update_weights_kernel(
        blocks_per_grid, 
        (threads_per_block,),
        (d_weights_delta, d_X_embedded, d_error, lr, 
         n_samples, embedding_dim, n_classes, tile_size),
        shared_mem=shared_mem_bytes
    )

    # Apply accumulated delta to weights
    d_weights += d_weights_delta

    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}/{epochs} complete.")

# --- Final weights to CPU ---
final_weights_embedded = d_weights.get()
print("Training finished.")

# Reconstruct full-space weights: W_full = V @ W_embedded
final_weights_full = embedding_matrix.dot(final_weights_embedded)

# --- Evaluation ---
print("\nEvaluating on CPU...")

# Option 1: Use embedded space
scores_embedded = X_embedded_cpu.dot(final_weights_embedded)

# Option 2: Use full space (equivalent but shows reconstruction)
scores_full = X_sparse_cpu.dot(final_weights_full)

# Use embedded space for evaluation
exp_scores = np.exp(scores_embedded - np.max(scores_embedded, axis=1, keepdims=True))
probabilities = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
predictions = np.argmax(probabilities, axis=1)

accuracy = np.mean(predictions == y_cpu)
print(f"\nSVD-Embedded Softmax Regression Final Accuracy: {accuracy * 100:.2f}%")
print(f"Embedding dimension: {embedding_dim} (reduced from {n_features})")

print("------------------------------------------\n")
