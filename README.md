# HPC_Project: GPU-Accelerated Financial Sentiment Analysis

## 1. Project Overview

This project presents a comprehensive framework for high-performance financial sentiment analysis using custom CUDA C++ kernels in a Python environment. The core objective is to move beyond standard library calls and implement the fundamental computational logic of various Machine Learning (ML) and Deep Learning (DL) models directly on the GPU.

The methodology is centered around a **Hybrid CUDA Development Model**. This pragmatic approach uses high-level, optimized CuPy for standard GPU operations (like dense matrix multiplication) while identifying the most unique or computationally intensive part of each algorithm and accelerating it with a custom, from-scratch CUDA C++ kernel.

### Dataset: Financial PhraseBank
The project utilizes the well-regarded Financial PhraseBank dataset, which consists of 4,840 English-language sentences from financial news articles. All sentences were labeled for sentiment (Positive, Negative, Neutral) by 16 finance and business experts, ensuring high-quality, domain-specific annotations.

---

## 2. Phase 1: Foundational Models with TF-IDF Embeddings

The initial phase focused on establishing strong performance baselines using classic ML and DL models. The text data was preprocessed using a standard TF-IDF vectorizer to create a high-dimensional (2000-feature) sparse representation of the sentences.

### Analysis of Foundational Models:

*   **Artificial Neural Networks (ANN):** The simple 1-hidden-layer ANN proved to be the most effective model on the TF-IDF features, achieving the highest accuracy. Its non-linear activation function allowed it to capture complex patterns that were inaccessible to the other models. The custom CUDA kernel was crucial for efficiently calculating the gradient of the first layer connected to the sparse input data (`X.T @ delta`).

*   **Support Vector Machine (SVM) with Conjugate Gradient:** This was the strongest-performing non-neural model. It demonstrates an advanced numerical method, solving for the optimal weights directly by using a CUDA-accelerated Conjugate Gradient solver. The custom CUDA kernels for sparse matrix-vector products were the core of this high-performance solver.

*   **Ensemble Models (Random Forest & Gradient Boosting):** These models showcased a different parallel programming paradigm. Instead of algebraic operations, their bottleneck is a combinatorial search for the best split point in a tree. A powerful CUDA kernel was written to parallelize this search, with thousands of threads evaluating potential splits simultaneously to find the one with the best Gini Impurity or MSE.

---

## 3. Phase 2: Advanced Architectures with Semantic Embeddings

Building on the insights from the foundational models, the project's next phase explores more advanced feature extraction techniques and state-of-the-art DL architectures, as outlined in the formal project plan.

### Advanced Feature Engineering & Modeling:

*   **Dimensionality Reduction (SVD & NMF):** To move beyond sparse features, techniques like Singular Value Decomposition (SVD) and Non-negative Matrix Factorization (NMF) are applied. These methods create dense, lower-dimensional feature representations. The matrix operations central to these algorithms are heavily accelerated using custom CUDA kernels for improved efficiency.

*   **Semantic Embeddings (Sentence-BERT):** To capture the contextual meaning of sentences—a weakness of TF-IDF—the project leverages a powerful, pretrained Sentence-BERT model. This generates high-quality, dense semantic embeddings for each sentence, providing a rich input for the final model.

*   **Transformer-Based Model Integration:** The final stage of the project integrates a Transformer-based architecture with the optimized Sentence-BERT embeddings. The core components of the Transformer, such as the self-attention mechanism, are composed of matrix multiplications and activation functions. Custom CUDA kernels, specifically designed with **tiling techniques** for efficient cache usage, are implemented to accelerate these large-scale computations, forming the heart of the high-performance model.

## 4. Custom CUDA Kernel Optimization

A key focus of this project was the development of custom, low-level CUDA C++ kernels for critical operations. This was done to maximize performance and demonstrate a fundamental understanding of GPU programming. Key implementations include:
*   **Optimized Matrix Multiplication:** Kernels designed for both sparse-to-dense and dense-dense matrix products.
*   **Tiling Techniques:** In the Transformer phase, shared memory and tiling are used to optimize matrix multiplication for large-scale computations, drastically reducing global memory access.
*   **Custom Activation Functions:** Efficient parallel implementations of Softmax and ReLU, including numerically stable versions.
*   **Parallel Reductions:** Advanced patterns like using `atomicMin` to find the best tree split across thousands of threads simultaneously.

## 5. Outcome & Future Work

### Outcome
This project successfully developed a GPU-accelerated framework for efficient financial text processing. By implementing custom CUDA kernels within a hybrid development model, we achieved significant reductions in computation time while building a suite of models with high prediction accuracy. The project is a definitive demonstration of integrating low-level CUDA programming with modern ML/DL pipelines.

### Future Work
*   **Kernel Optimization:** Further optimize the custom CUDA kernels for enhanced scalability, particularly for the Transformer's attention mechanism.
*   **Advanced Architectures:** Experiment with additional embeddings (e.g., GloVe, FastText) and more advanced deep learning architectures like Long-Formers or BigBird for handling longer financial documents.
*   **Real-Time Analysis:** Explore the deployment of the final Transformer model in a real-time pipeline for live sentiment prediction from financial news feeds.

## 6. Technologies Used

*   **Programming Language:** Python
*   **GPU Acceleration:** CUDA C++, CuPy
*   **Data Processing:** NumPy, Pandas, Scikit-learn
*   **Semantic Embeddings:** Sentence-BERT
