
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import TruncatedSVD
import cupy as cp
import time

# Data Preprocessing + SVD Embedding
df = pd.read_csv('fin_data_1.csv')
df.columns = ['Sentence', 'Sentiment']
df.dropna(subset=['Sentence', 'Sentiment'], inplace=True)
df = df[df['Sentence'].apply(lambda x: isinstance(x, str))]
df['Clean_Sentence'] = df['Sentence'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x.lower()))

# TF-IDF
vectorizer = TfidfVectorizer(max_features=2000, stop_words='english')
X_tfidf = vectorizer.fit_transform(df['Clean_Sentence'])

# SVD Embedding
svd_components = 256
svd = TruncatedSVD(n_components=svd_components, random_state=42)
X_svd = svd.fit_transform(X_tfidf).astype(np.float32)

# Encode labels
le = LabelEncoder()
y = le.fit_transform(df['Sentiment'])
n_samples, n_features = X_svd.shape
n_classes = len(np.unique(y))

print(f"Data shape after SVD: {X_svd.shape}")

# GPU Setup
X_gpu = cp.asarray(X_svd)
y_gpu = cp.asarray(np.eye(n_classes)[y], dtype=cp.float32)

h1_size, h2_size = 128, 64
lr, epochs = 0.05, 1500

# CUDA Kernel for Dense Weight Update
dense_update_kernel_code = r'''
extern "C" __global__
void dense_update_w1(
    const float* __restrict__ X,
    const float* __restrict__ delta_h1,
    float* __restrict__ w1,
    float* __restrict__ b1,
    float lr, int n_samples, int n_features, int h1_size)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < n_features && row < h1_size) {
        float grad = 0.0f;
        for (int i = 0; i < n_samples; ++i)
            grad += X[i * n_features + col] * delta_h1[i * h1_size + row];
        w1[col * h1_size + row] -= lr * grad / n_samples;
    }

    if (col == 0 && row < h1_size) {
        float bgrad = 0.0f;
        for (int i = 0; i < n_samples; ++i)
            bgrad += delta_h1[i * h1_size + row];
        b1[row] -= lr * bgrad / n_samples;
    }
}
'''
dense_update_kernel = cp.RawKernel(dense_update_kernel_code, 'dense_update_w1')

# Initialize Weights
w1 = (cp.random.randn(n_features, h1_size) * cp.sqrt(2. / n_features)).astype(cp.float32)
b1 = cp.zeros(h1_size, dtype=cp.float32)
w2 = (cp.random.randn(h1_size, h2_size) * cp.sqrt(2. / h1_size)).astype(cp.float32)
b2 = cp.zeros(h2_size, dtype=cp.float32)
w3 = (cp.random.randn(h2_size, n_classes) * cp.sqrt(2. / h2_size)).astype(cp.float32)
b3 = cp.zeros(n_classes, dtype=cp.float32)

threads = (16, 16)
blocks = ((n_features + threads[0] - 1) // threads[0],
          (h1_size + threads[1] - 1) // threads[1])

# Training Loop
print(f"Network: {n_features}->{h1_size}->{h2_size}->{n_classes}. Training...")

start = time.time()

for epoch in range(epochs):
    h1_raw = X_gpu.dot(w1) + b1
    h1_act = cp.maximum(0, h1_raw)
    h2_raw = h1_act.dot(w2) + b2
    h2_act = cp.maximum(0, h2_raw)
    scores = h2_act.dot(w3) + b3
    exp_scores = cp.exp(scores - scores.max(axis=1, keepdims=True))
    probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)

    delta_out = (probs - y_gpu)
    delta_h2 = delta_out.dot(w3.T) * (h2_raw > 0)
    delta_h1 = delta_h2.dot(w2.T) * (h1_raw > 0)

    w3 -= lr * (h2_act.T.dot(delta_out) / n_samples)
    b3 -= lr * (cp.sum(delta_out, axis=0) / n_samples)
    w2 -= lr * (h1_act.T.dot(delta_h2) / n_samples)
    b2 -= lr * (cp.sum(delta_h2, axis=0) / n_samples)

    dense_update_kernel(
        blocks, threads,
        (X_gpu, delta_h1, w1, b1,
         lr, n_samples, n_features, h1_size)
    )

    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}/{epochs} complete.")

end = time.time()

# Evaluation
print("Training finished.")

h1 = np.maximum(0, X_svd.dot(cp.asnumpy(w1)) + cp.asnumpy(b1))
h2 = np.maximum(0, h1.dot(cp.asnumpy(w2)) + cp.asnumpy(b2))
scores = h2.dot(cp.asnumpy(w3)) + cp.asnumpy(b3)
exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
preds = np.argmax(probs, axis=1)
acc = np.mean(preds == y)
print(f"Runtime: {end - start:.3f}s | Accuracy: {acc*100:.2f}%")
print("------------------------------------------\n")
