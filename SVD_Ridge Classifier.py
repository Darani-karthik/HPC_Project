
import pandas as pd
import numpy as np
import re
import cupy as cp
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Load and clean data
df = pd.read_csv('fin_data_1.csv')
df.columns = ['Sentence', 'Sentiment']
df.dropna(subset=['Sentence', 'Sentiment'], inplace=True)
df = df[df['Sentence'].apply(lambda x: isinstance(x, str))]
df['Clean_Sentence'] = df['Sentence'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x.lower()))

# TF-IDF
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
X_tfidf = tfidf.fit_transform(df['Clean_Sentence'])

# SVD embedding (dense)
svd = TruncatedSVD(n_components=512, random_state=42)
X_svd = svd.fit_transform(X_tfidf)

# Labels
le = LabelEncoder()
y = le.fit_transform(df['Sentiment'])
classes = np.unique(y)
n_classes = len(classes)

print(f"SVD embedding shape: {X_svd.shape}")
print("Classes:", dict(zip(le.classes_, range(n_classes))))

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_svd, y, test_size=0.2, random_state=42, stratify=y
)

# Move to GPU
X_train_gpu = cp.asarray(X_train.astype(np.float32))
y_train_gpu = cp.asarray(y_train.astype(np.int32))
n_samples, n_features = X_train_gpu.shape

print(f"Training samples: {n_samples}, features: {n_features}")

# CUDA Kernels for Dense Matrix Operations
dense_mv_kernel = r'''
extern "C" __global__
void dense_mv(const float* X, const float* vec, float* out, int n_samples, int n_features) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_samples) return;
    float sum = 0.0f;
    for (int j = 0; j < n_features; ++j) {
        sum += X[i * n_features + j] * vec[j];
    }
    out[i] = sum;
}
'''

dense_mtv_kernel = r'''
extern "C" __global__
void dense_mtv(const float* X, const float* vec, float* out, int n_samples, int n_features) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_features) return;
    float sum = 0.0f;
    for (int i = 0; i < n_samples; ++i) {
        sum += X[i * n_features + j] * vec[i];
    }
    out[j] = sum;
}
'''

dense_mv = cp.RawKernel(dense_mv_kernel, 'dense_mv')
dense_mtv = cp.RawKernel(dense_mtv_kernel, 'dense_mtv')

threads_per_block = 256

# Wrapper functions
def gpu_X_dot_vec(X, vec):
    n_samples, n_features = X.shape
    out = cp.zeros(n_samples, dtype=cp.float32)
    blocks = (n_samples + threads_per_block - 1) // threads_per_block
    dense_mv((blocks,), (threads_per_block,), (X, vec, out, n_samples, n_features))
    return out

def gpu_XT_dot_vec(X, vec):
    n_samples, n_features = X.shape
    out = cp.zeros(n_features, dtype=cp.float32)
    blocks = (n_features + threads_per_block - 1) // threads_per_block
    dense_mtv((blocks,), (threads_per_block,), (X, vec, out, n_samples, n_features))
    return out

# Conjugate Gradient Ridge Solver

def A_dot(X, v, alpha=1.0):
    tmp = gpu_X_dot_vec(X, v)
    out = gpu_XT_dot_vec(X, tmp)
    if alpha != 0.0:
        out += alpha * v
    return out

def cg_solve(X, b, alpha=1.0, max_iter=1000, tol=1e-5):
    x = cp.zeros_like(b)
    r = b - A_dot(X, x, alpha)
    p = r.copy()
    rsold = float(cp.vdot(r, r).real)
    for i in range(max_iter):
        Ap = A_dot(X, p, alpha)
        alpha_cg = rsold / (float(cp.vdot(p, Ap).real) + 1e-30)
        x += alpha_cg * p
        r -= alpha_cg * Ap
        rsnew = float(cp.vdot(r, r).real)
        if rsnew < tol * tol:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew
    return x

# Multi-class Training 

alpha = 1.0
W = np.zeros((n_features, n_classes), dtype=np.float32)
bias = np.zeros(n_classes, dtype=np.float32)

start = time.time()
for ci, c in enumerate(classes):
    print(f"Training class {ci+1}/{n_classes}...")
    y_bin = np.where(y_train == c, 1.0, -1.0).astype(np.float32)
    d_y = cp.asarray(y_bin)
    b_vec = gpu_XT_dot_vec(X_train_gpu, d_y)
    w_sol = cg_solve(X_train_gpu, b_vec, alpha=alpha, max_iter=500, tol=1e-5)
    w_cpu = cp.asnumpy(w_sol)
    W[:, ci] = w_cpu
    resid = y_bin - X_train @ w_cpu
    bias[ci] = float(np.mean(resid))
end = time.time()

# Evaluation

train_scores = X_train @ W + bias
train_preds = np.argmax(train_scores, axis=1)
train_acc = accuracy_score(y_train, train_preds)

test_scores = X_test @ W + bias
test_preds = np.argmax(test_scores, axis=1)
test_acc = accuracy_score(y_test, test_preds)

print(f"\nTraining time: {end - start:.3f} s")
print(f"Train Accuracy: {train_acc*100:.2f}%")
print(f"Test Accuracy:  {test_acc*100:.2f}%")
