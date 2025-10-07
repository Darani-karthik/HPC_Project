# Imports
import numpy as np
import scipy.sparse
import cupy as cp
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time

# Load data
X_csr_full = scipy.sparse.load_npz('/content/preprocessed_features.npz').tocsr()
y_full = np.load('/content/preprocessed_labels.npy')

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_csr_full, y_full, test_size=0.2, random_state=42, stratify=y_full
)

# Data shapes and classes
n_samples, n_features = X_train.shape
classes = np.unique(y_train)
n_classes = len(classes)
print(f"Training on {n_samples} samples with {n_features} features.")
print(f"Number of classes: {n_classes}")

# Move CSR arrays to GPU
d_X_data = cp.asarray(X_train.data.astype(np.float32))
d_X_idx  = cp.asarray(X_train.indices.astype(np.int32))
d_X_ptr  = cp.asarray(X_train.indptr.astype(np.int32))

# CSR matrix-vector kernel
csr_mv_vector_src = r'''
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

extern "C" __global__
void csr_mv_vector(const float* Xd, const int* Xidx, const int* Xptr,
                   const float* vec, float* out, int n_samples) {
    int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = global_thread_id / warpSize; 
    int lane = global_thread_id % warpSize;   
    if (warp_id >= n_samples) return;
    int row_start = Xptr[warp_id];
    int row_end   = Xptr[warp_id+1];
    float sum = 0.0f;
    for (int p = row_start + lane; p < row_end; p += warpSize) {
        sum += Xd[p] * vec[Xidx[p]];
    }
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(cg::this_thread_block());
    for (int offset = warp.size() / 2; offset > 0; offset /= 2) {
        sum += warp.shfl_down(sum, offset);
    }
    if (lane == 0) out[warp_id] = sum;
}
'''

# CSR^T * vector kernel
csr_mvt_improved_src = r'''
extern "C" __global__
void csr_mvt_improved(const float* Xd, const int* Xidx, const int* Xptr,
                      const float* invec, float* out, int n_samples) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_samples) return;
    float v = invec[i];
    for (int p = Xptr[i]; p < Xptr[i+1]; ++p) {
        atomicAdd(&out[Xidx[p]], Xd[p] * v);
    }
}
'''

# Compile and wrap kernels
csr_mv_vector = cp.RawKernel(csr_mv_vector_src, 'csr_mv_vector')
csr_mvt_improved = cp.RawKernel(csr_mvt_improved_src, 'csr_mvt_improved')

# Kernel launch parameters
threads_per_block = 256

# Wrapper: X * vec
def gpu_X_dot_vec(vec):
    out = cp.zeros(n_samples, dtype=cp.float32)
    total_threads = n_samples * 32
    blocks = (total_threads + threads_per_block - 1) // threads_per_block
    csr_mv_vector((blocks,), (threads_per_block,), (d_X_data, d_X_idx, d_X_ptr, vec, out, n_samples))
    return out

# Wrapper: X^T * vec
def gpu_XT_dot_vec(invec):
    out = cp.zeros(n_features, dtype=cp.float32)
    blocks = (n_samples + threads_per_block - 1) // threads_per_block
    csr_mvt_improved((blocks,), (threads_per_block,), (d_X_data, d_X_idx, d_X_ptr, invec, out, n_samples))
    return out

# Linear operator (A * v)
def A_dot(v, alpha=1.0):
    tmp = gpu_X_dot_vec(v)
    out = gpu_XT_dot_vec(tmp)
    if alpha != 0.0:
        out += alpha * v
    return out

# Conjugate Gradient solver
def cg_solve(b, alpha=1.0, max_iter=1000, tol=1e-6):
    x = cp.zeros_like(b)
    r = b - A_dot(x, alpha)
    p = r.copy()
    rsold = float(cp.vdot(r, r).real)
    if rsold < tol * tol:
        return x
    for i in range(max_iter):
        Ap = A_dot(p, alpha)
        alpha_cg = rsold / (float(cp.vdot(p, Ap).real) + 1e-30)
        x += alpha_cg * p
        r -= alpha_cg * Ap
        rsnew = float(cp.vdot(r, r).real)
        if rsnew < tol * tol:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew
    return x

# Training loop
alpha = 1.0
W = np.zeros((n_features, n_classes), dtype=np.float32)
bias = np.zeros(n_classes, dtype=np.float32)

start = time.time()
for ci, c in enumerate(classes):
    print(f"Solving class {ci+1}/{n_classes} (label={c})")
    y_bin = np.where(y_train == c, 1.0, -1.0).astype(np.float32)
    d_y = cp.asarray(y_bin)
    b_vec = gpu_XT_dot_vec(d_y)
    w_sol = cg_solve(b_vec, alpha=alpha, max_iter=500, tol=1e-5)
    w_cpu = cp.asnumpy(w_sol)
    W[:, ci] = w_cpu
    resid = y_bin - X_train.dot(w_cpu)
    bias[ci] = float(np.mean(resid))
end = time.time()

# Evaluation
train_scores = X_train.dot(W) + bias
train_preds = classes[np.argmax(train_scores, axis=1)]
train_acc = accuracy_score(y_train, train_preds)

test_scores = X_test.dot(W) + bias
test_preds = classes[np.argmax(test_scores, axis=1)]
test_acc = accuracy_score(y_test, test_preds)

print(f"\nBlock runtime: {end - start:.3f} seconds")
print(f"Train accuracy: {train_acc*100:.2f}%")
print(f"Test accuracy: {test_acc*100:.2f}%")
