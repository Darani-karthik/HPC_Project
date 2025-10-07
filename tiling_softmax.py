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

# --- Tiled CUDA kernel with shared memory ---
update_weights_tiled_kernel_code = r'''
extern "C" _global_
void update_weights_tiled_kernel(float* _restrict_ weights,
                                  const float* _restrict_ X_data,
                                  const int* _restrict_ X_indices,
                                  const int* _restrict_ X_indptr,
                                  const float* _restrict_ error,
                                  float lr,
                                  int n_samples,
                                  int n_features,
                                  int n_classes,
                                  int tile_size) {

    int sample_idx = blockIdx.x;
    if (sample_idx >= n_samples) return;

    int tid = threadIdx.x;

    // Shared memory for tiling error values
    extern _shared_ float shared_error[];

    int start = X_indptr[sample_idx];
    int end = X_indptr[sample_idx + 1];

    const float* err_row = &error[sample_idx * n_classes];
    
    float scale = - (lr / (float)n_samples);

    // Process classes in tiles to reuse error values from shared memory
    for (int tile_start = 0; tile_start < n_classes; tile_start += tile_size) {
        int tile_end = min(tile_start + tile_size, n_classes);
        int tile_actual_size = tile_end - tile_start;
        
        // Load error tile into shared memory (coalesced read)
        for (int i = tid; i < tile_actual_size; i += blockDim.x) {
            shared_error[i] = err_row[tile_start + i];
        }
        __syncthreads();

        // Process all non-zero features for this sample against this tile
        for (int idx = start; idx < end; ++idx) {
            int feat = X_indices[idx];
            float x = X_data[idx];

            // Update weights for all classes in current tile
            for (int i = tid; i < tile_actual_size; i += blockDim.x) {
                int class_idx = tile_start + i;
                float g = x * shared_error[i];  // Use shared memory
                float delta = scale * g;
                atomicAdd(&weights[feat * n_classes + class_idx], delta);
            }
        }
        __syncthreads();
    }
}
'''

# --- Alternative: Feature-parallel tiled kernel ---
# This version parallelizes over features instead of classes
update_weights_feature_tiled_kernel_code = r'''
extern "C" _global_
void update_weights_feature_tiled_kernel(float* _restrict_ weights,
                                          const float* _restrict_ X_data,
                                          const int* _restrict_ X_indices,
                                          const int* _restrict_ X_indptr,
                                          const float* _restrict_ error,
                                          float lr,
                                          int n_samples,
                                          int n_features,
                                          int n_classes,
                                          int tile_size) {

    int sample_idx = blockIdx.x;
    if (sample_idx >= n_samples) return;

    int tid = threadIdx.x;
    int block_size = blockDim.x;

    // Shared memory for feature values and indices
    extern _shared_ char shared_mem[];
    float* shared_x = (float*)shared_mem;
    int* shared_feat = (int*)&shared_x[tile_size];
    
    int start = X_indptr[sample_idx];
    int end = X_indptr[sample_idx + 1];
    int nnz = end - start;

    const float* err_row = &error[sample_idx * n_classes];
    float scale = - (lr / (float)n_samples);

    // Process non-zero features in tiles
    for (int tile_start = start; tile_start < end; tile_start += tile_size) {
        int tile_end = min(tile_start + tile_size, end);
        int tile_actual_size = tile_end - tile_start;
        
        // Load feature tile into shared memory
        for (int i = tid; i < tile_actual_size; i += block_size) {
            int idx = tile_start + i;
            shared_x[i] = X_data[idx];
            shared_feat[i] = X_indices[idx];
        }
        __syncthreads();

        // Each thread processes some features from the tile
        for (int i = tid; i < tile_actual_size; i += block_size) {
            float x = shared_x[i];
            int feat = shared_feat[i];

            // Update all classes for this feature
            for (int class_idx = 0; class_idx < n_classes; ++class_idx) {
                float g = x * err_row[class_idx];
                float delta = scale * g;
                atomicAdd(&weights[feat * n_classes + class_idx], delta);
            }
        }
        __syncthreads();
    }
}
'''

# Choose which kernel to use
USE_CLASS_TILED = True  # Set to False for feature-tiled version

if USE_CLASS_TILED:
    update_weights_kernel = cp.RawKernel(update_weights_tiled_kernel_code, 
                                         'update_weights_tiled_kernel')
    kernel_name = "Class-Tiled"
    # Tile size for classes (error values)
    tile_size = min(256, n_classes)
    threads_per_block = min(256, max(32, n_classes))
    shared_mem_bytes = tile_size * 4  # 4 bytes per float
else:
    update_weights_kernel = cp.RawKernel(update_weights_feature_tiled_kernel_code, 
                                         'update_weights_feature_tiled_kernel')
    kernel_name = "Feature-Tiled"
    # Tile size for features
    avg_nnz = X_sparse_cpu.nnz / n_samples
    tile_size = min(128, int(avg_nnz))
    threads_per_block = 128
    shared_mem_bytes = tile_size * (4 + 4)  # floats + ints

blocks_per_grid = (n_samples,)

print(f"Kernel: {kernel_name}")
print(f"Tile size: {tile_size}")
print(f"Threads per block: {threads_per_block}")
print(f"Shared memory per block: {shared_mem_bytes} bytes")

# Preallocate intermediate arrays
d_scores = cp.empty((n_samples, n_classes), dtype=cp.float32)
exp_scores = cp.empty_like(d_scores)
d_probabilities = cp.empty_like(d_scores)
d_error = cp.empty_like(d_scores)

print("\nStarting training (tiled custom kernel)...")
for epoch in range(epochs):
    # forward
    d_scores = d_X_sparse.dot(d_weights)
    
    # stable softmax
    row_max = d_scores.max(axis=1, keepdims=True)
    exp_scores = cp.exp(d_scores - row_max)
    d_probabilities = exp_scores / exp_scores.sum(axis=1, keepdims=True)

    # error (prob - label)
    d_error = d_probabilities - d_y_one_hot

    # compute delta updates with tiled kernel
    d_weights_delta = cp.zeros_like(d_weights)
    update_weights_kernel(
        blocks_per_grid, 
        (threads_per_block,),
        (d_weights_delta, d_X_data, d_X_indices, d_X_indptr,
         d_error, lr, n_samples, n_features, n_classes, tile_size),
        shared_mem=shared_mem_bytes
    )

    # apply accumulated delta to weights
    d_weights += d_weights_delta

    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}/{epochs} complete.")

# --- Final weights to CPU ---
final_weights = d_weights.get()
print("Training finished.")

# --- Evaluation  ---
print("Evaluating on CPU...")

scores = X_sparse_cpu.dot(final_weights)
exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
probabilities = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
predictions = np.argmax(probabilities, axis=1)

accuracy = np.mean(predictions == y_cpu)
print(f"\n{kernel_name} CUDA Kernel Final Accuracy: {accuracy * 100:.2f}%")

print("------------------------------------------\n")