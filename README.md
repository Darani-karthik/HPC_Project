# HPC_Project: GPU-Accelerated Financial Sentiment Analysis

This project explores the implementation of **Machine Learning (ML)** and **Deep Learning (DL)** models using **CUDA programming in Python**. Instead of relying solely on high-level libraries, the core objective is to leverage low-level CUDA kernels to perform data preprocessing, training, and inference on the GPU.

---

## Dataset

The project uses the **Financial PhraseBank** dataset:

- **Total samples:** 4,840 English-language sentences  
- **Source:** Financial news articles  
- **Labels:** Positive, Negative, Neutral  
- **Annotation:** Labeled by 16 finance and business experts to ensure domain-specific accuracy

---

## Methodology

### 1) Data Preprocessing
- Applied **One-Hot Encoding (OHE)**, **Singular Value Decomposition (SVD)**, and **Non-negative Matrix Factorization (NMF)** for feature extraction and dimensionality reduction  
- Cleaned, tokenized, and transformed data into embeddings suitable for ML and DL models

### 2) CUDA-Based Optimization
- Implemented custom **CUDA kernels** to parallelize computationally intensive tasks  
- Optimized **matrix multiplication**, **tiling techniques**, and efficient **Softmax** and **ReLU** activation functions

### 3) Modeling
- Traditional ML models for baseline evaluation  
- Transformer-based architectures integrated with optimized embeddings for deep learning

---

## Pipeline Steps

### Step 1 – One-Hot Encoding (OHE)
- Generated OHE embeddings for categorical features  
- Trained a baseline transformer using these embeddings on GPU

### Step 2 – Handling Imbalanced Data with SMOTE
- Applied **SMOTE** to oversample minority classes  
- Ensured fair representation of all classes in embeddings

### Step 3 – TF-IDF Feature Extraction
- Generated **TF-IDF embeddings** from textual data  
- Captured term frequency and importance for richer representations

### Step 4 – Singular Value Decomposition (SVD)
- Applied **SVD** for dimensionality reduction on TF-IDF and other embeddings  
- Accelerated matrix operations using CUDA  
- Monitored negative embeddings generation and its impact on downstream models

### Step 5 – Non-negative Matrix Factorization (NMF)
- Applied **NMF** to extract non-negative features  
- Optimized CUDA kernels to handle large datasets efficiently  
- Evaluated effects on deep learning model performance

### Step 6 – Custom CUDA Kernel Optimization & Tiling
- Developed optimized **CUDA kernels** for matrix multiplication, activations, and Softmax  
- Implemented **tiling techniques** for large-scale matrix computations  
- Incorporated **MMA** (Matrix Multiply-Accumulate) for high throughput training

### Step 7 – Sentence-BERT Embedding Generation
- Used pretrained **Sentence-BERT** to generate dense semantic embeddings  
- Fine-tuned embeddings for task-specific classification or regression

### Step 8 – selection of the best embedding techinque & Transformer Integration
- All embedding techniques—including OHE, TF-IDF, SVD/NMF, and Sentence-BERT—were experimented with and integrated into a transformer-based model, leveraging GPU acceleration and optimized CUDA kernels to achieve faster and more efficient training.

### Step 9 – Model Training and Evaluation
- Trained transformer using fused embeddings  
- Evaluated model performance using metrics:  
  - Accuracy  
  - F1-score  
  - Precision  
  - Recall  
  - Robustness on balanced datasets

---
## Technologies Used

- **Python** – Primary programming language for data processing, model development, and CUDA integration  
- **CUDA** – GPU acceleration for custom kernels, matrix operations, and high-performance training  
- **NumPy** – Efficient numerical computations and matrix operations  
- **Pandas** – Data loading, preprocessing, and manipulation  
- **Scikit-learn** – Traditional ML models, preprocessing, SVD, NMF, and evaluation metrics  
- **Torch (PyTorch)** – Deep learning framework for transformer models and GPU training  
- **Sentence-BERT** – Pretrained model for generating dense semantic embeddings from financial text  
- **CuPy** – GPU-accelerated array computations compatible with NumPy  
- **Imbalanced-learn** – Handling class imbalance using SMOTE and other resampling techniques
