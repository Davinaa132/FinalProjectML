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

# Pastikan path file CSV relatif atau lokal
data_path = 'Scrapping.csv'  # Disarankan: folder `data/` dalam repo
model_dir = 'model'
os.makedirs(model_dir, exist_ok=True)

# Membaca data
df = pd.read_csv(data_path, encoding='ISO-8859-1', engine='python', on_bad_lines='warn', delimiter=';')

# Gabungkan kolom headline + body
df['text'] = df['Headline'].astype(str) + " " + df['Body'].astype(str)

# Split fitur dan label
X = df['text']
y = df['Label']

# Bagi data latih dan uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Stopword Indonesia
factory = StopWordRemoverFactory()
indonesian_stop_words = factory.get_stop_words()

# TF-IDF
vectorizer = TfidfVectorizer(stop_words=indonesian_stop_words, max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)

# Latih model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Simpan model dan vectorizer
model_path = os.path.join(model_dir, 'multinomial_nb_model.pkl')
vectorizer_path = os.path.join(model_dir, 'tfidf_vectorizer.pkl')

with open(model_path, 'wb') as f:
    pickle.dump(model, f)

with open(vectorizer_path, 'wb') as f:
    pickle.dump(vectorizer, f)

print("✅ Model dan vectorizer berhasil disimpan ke folder 'model/'")
