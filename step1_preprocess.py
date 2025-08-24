# step1_preprocess.py

import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import scipy.sparse

def preprocess_and_save(input_path='fin_data_1.csv'):
    """
    Loads, preprocesses financial sentiment data, and saves it for model training.
    """
    print("--- Running Step 1: Data Preprocessing ---")
    
    # Load the dataset
    try:
        df = pd.read_csv(input_path)
        df.columns = ['Sentence', 'Sentiment']
    except FileNotFoundError:
        print(f"Error: The input file '{input_path}' was not found in the current directory.")
        return

    # --- Text Cleaning ---
    df.dropna(subset=['Sentence', 'Sentiment'], inplace=True)
    df = df[df['Sentence'].apply(lambda x: isinstance(x, str))]
    df['Clean_Sentence'] = df['Sentence'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x.lower()))

    # --- Feature Engineering (TF-IDF) ---
    vectorizer = TfidfVectorizer(max_features=2000, stop_words='english')
    X_sparse = vectorizer.fit_transform(df['Clean_Sentence'])
    
    # --- Label Encoding ---
    le = LabelEncoder()
    y = le.fit_transform(df['Sentiment'])
    
    print("Sentiment classes mapped to integers:")
    for i, class_name in enumerate(le.classes_):
        print(f"- {class_name}: {i}")

    # --- Save Preprocessed Data ---
    scipy.sparse.save_npz('preprocessed_features.npz', X_sparse)
    np.save('preprocessed_labels.npy', y)
    
    print("\nPreprocessing complete.")
    print(f"Features shape: {X_sparse.shape}")
    print(f"Labels shape: {y.shape}")
    print("Preprocessed data saved to 'preprocessed_features.npz' and 'preprocessed_labels.npy'.")
    print("------------------------------------------\n")

if __name__ == '__main__':
    preprocess_and_save()