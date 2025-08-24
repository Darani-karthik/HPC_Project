# HPC_Project: GPU-Accelerated Financial Sentiment Analysis

This project explores the implementation of **Machine Learning (ML)** and **Deep Learning (DL)** models using **CUDA programming in Python**. Instead of relying solely on high-level libraries, the core objective is to leverage low-level CUDA kernels to perform data preprocessing, training, and inference on the GPU.

---

## Dataset

The project uses the **Financial PhraseBank** dataset:

- **Total samples:** 4,840 English-language sentences.
- **Source:** Financial news articles.
- **Labels:** Positive, Negative, Neutral.
- **Annotation:** Labeled by 16 finance and business experts to ensure domain-specific accuracy.

---

## Methodology

### 1) Data Preprocessing

- Applied **One-Hot Encoding (OHE)**, **Singular Value Decomposition (SVD)**, and **Non-negative Matrix Factorization (NMF)** for feature extraction and dimensionality reduction.
- Cleaned, tokenized, and transformed data into embeddings suitable for ML and DL models.

### 2) CUDA-Based Optimization

- Implemented custom **CUDA kernels** to parallelize computationally intensive tasks.
- Optimized **matrix multiplication**, **tiling techniques**, and efficient **Softmax** and **ReLU** activation functions.

### 3) Modeling

- Traditional ML models for baseline evaluation.
- Transformer-based architectures integrated with optimized embeddings for deep learning.

---

## Phased Implementation

### Phase 1 – One-Hot Encoding (OHE)
- Generated OHE embeddings.
- Trained traditional transformer  and tested performance on GPU.

### Phase 2 – Singular Value Decomposition (SVD)
- Reduced feature dimensionality using SVD.
- Accelerated matrix operations with CUDA for better efficiency.
- observed negative embeddings generation
  
### Phase 3 – Non-negative Matrix Factorization (NMF)
- Applied NMF for feature extraction.
- Assessed impact on deep learning model accuracy.
- Optimized CUDA kernels for larger datasets.

### Phase 4 – Custom CUDA Kernel Optimization
- Developed CUDA kernels for **matrix multiplication**, **activations**, and **Softmax**.
- Implemented **tiling techniques** for large-scale computations.

### Phase 5 – Pretrained Sentence-BERT Embeddings
- Used **Sentence-BERT** for generating dense embeddings.
- Fine-tuned model for improved sentiment analysis.

### Phase 6 – Transformer-Based Model Integration
- Integrated **Transformer architectures** with optimized embeddings.
- Tested and fine-tuned for high prediction accuracy.

---

## Future Work

- Further optimize CUDA kernels for enhanced scalability and efficiency.
- Experiment with additional embeddings and advanced deep learning architectures.
- Explore real-time financial news analysis for live sentiment prediction.

---

## Outcome

- Developed a **GPU-accelerated framework** for efficient financial text processing.
- Achieved **significant reductions in training and inference time** while maintaining or improving sentiment prediction accuracy.
- Demonstrated integration of **CUDA kernels with ML/DL pipelines**.

---

## Technologies Used

- **Python**  
- **CUDA** for GPU acceleration  
- **NumPy, Pandas** for data processing  
- **Scikit-learn** for ML models  
- **PyTorch** for deep learning and Transformer architectures  
- **Sentence-BERT** for semantic embeddings  

---

## License

This project is released under the **MIT License**.
