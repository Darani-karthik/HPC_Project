# Imports
import numpy as np
import scipy.sparse
import cupy as cp
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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

# Move CSR arrays to GPU
d_X_data = cp.asarray(X_train.data.astype(np.float32))
d_X_idx  = cp.asarray(X_train.indices.astype(np.int32))
d_X_ptr  = cp.asarray(X_train.indptr.astype(np.int32))

# GPU Kernels for CSR matrix-vector ops
csr_mv_src = r'''
extern "C" __global__
void csr_mv(const float* Xd, const int* Xidx, const int* Xptr,
            const float* vec, float* out, int n_samples) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i >= n_samples) return;
  float s = 0.0f;
  for (int p = Xptr[i]; p < Xptr[i+1]; ++p) {
    s += Xd[p] * vec[Xidx[p]];
  }
  out[i] = s;
}
'''
csr_mvt_src = r'''
extern "C" __global__
void csr_mvt(const float* Xd, const int* Xidx, const int* Xptr,
             const float* invec, float* out, int n_samples) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i >= n_samples) return;
  float v = invec[i];
  if (v == 0.0f) return;
  for (int p = Xptr[i]; p < Xptr[i+1]; ++p) {
    atomicAdd(&out[Xidx[p]], Xd[p] * v);
  }
}
'''

# Kernel wrappers
csr_mv = cp.RawKernel(csr_mv_src, 'csr_mv')
csr_mvt = cp.RawKernel(csr_mvt_src, 'csr_mvt')

tp = 256
tp = 256 
blocks = (n_samples + tp - 1) // tp

def gpu_X_dot_vec(vec):
    out = cp.zeros(n_samples, dtype=cp.float32)
    csr_mv((blocks,), (tp,), (d_X_data, d_X_idx, d_X_ptr, vec, out, n_samples))
    return out

def gpu_XT_dot_vec(invec):
    out = cp.zeros(n_features, dtype=cp.float32)
    csr_mvt((blocks,), (tp,), (d_X_data, d_X_idx, d_X_ptr, invec, out, n_samples))
    return out

def A_dot(v, alpha=1.0):
    tmp = gpu_X_dot_vec(v)
    out = gpu_XT_dot_vec(tmp)
    if alpha != 0.0:
        out += alpha * v
    return out

def cg_solve(b, alpha=1.0, max_iter=1000, tol=1e-6):
    x = cp.zeros_like(b)
    r = b - A_dot(x, alpha)
    p = r.copy()
    rsold = float(cp.vdot(r, r).real)
    if rsold < tol*tol:
        return x
    for _ in range(max_iter):
        Ap = A_dot(p, alpha)
        alpha_cg = rsold / (float(cp.vdot(p, Ap).real) + 1e-30)
        x += alpha_cg * p
        r -= alpha_cg * Ap
        rsnew = float(cp.vdot(r, r).real)
        if rsnew < tol*tol:
            break
        p = r + (rsnew/rsold) * p
        rsold = rsnew
    return x

alpha = 1.0
W = np.zeros((n_features, n_classes), dtype=np.float32)
bias = np.zeros(n_classes, dtype=np.float32)

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

# Evaluate
train_scores = X_train.dot(W) + bias
train_preds = np.argmax(train_scores, axis=1)
train_acc = accuracy_score(y_train, train_preds)

test_scores = X_test.dot(W) + bias
test_preds = np.argmax(test_scores, axis=1)
test_acc = accuracy_score(y_test, test_preds)

print(f"\nTrain accuracy: {train_acc*100:.2f}%")
print(f"Test accuracy: {test_acc*100:.2f}%")

