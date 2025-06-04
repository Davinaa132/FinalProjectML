# -*- coding: utf-8 -*-
"""train_model.py

Script untuk melatih model deteksi hoaks dan menyimpan model serta vectorizer.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
import os
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# Path dataset dan folder output model
data_path = 'datasetUMPOHoax.csv'  # ← Ganti ke file baru
model_dir = 'model'
os.makedirs(model_dir, exist_ok=True)

# Baca dataset
df = pd.read_csv(data_path, encoding='ISO-8859-1', engine='python', on_bad_lines='warn')

# Pastikan kolom yang dibutuhkan ada
if 'tweet' not in df.columns or 'label' not in df.columns:
    raise ValueError("Dataset harus memiliki kolom 'tweet' dan 'label'.")

# Ambil fitur dan label
X = df['tweet'].astype(str)
y = df['label']

# Split train-test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Stopword bahasa Indonesia dari Sastrawi
factory = StopWordRemoverFactory()
indonesian_stop_words = factory.get_stop_words()

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words=indonesian_stop_words, max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)

# Latih model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Simpan model dan vectorizer
model_path = '/content/drive/MyDrive/Final_Project_ML/multinomial_nb_modelUMPOH.pkl'
vectorizer_path = '/content/drive/MyDrive/Final_Project_ML/tfidf_vectorizerUMPOH.pkl'

with open(model_path, 'wb') as f:
    pickle.dump(model, f)

with open(vectorizer_path, 'wb') as f:
    pickle.dump(vectorizer, f)

print("✅ Model dan vectorizer berhasil disimpan ke folder 'model/'")
