# -*- coding: utf-8 -*-
"""retrain_model.py

Script untuk retrain model deteksi hoaks menggunakan dataset awal + laporan kesalahan dari pengguna.
"""

import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# Path
dataset_asli = 'datasetUMPOHoax.csv'
laporan_path = 'laporan_kesalahan.csv'
model_path = 'multinomial_nb_modelUMPOH.pkl'
vectorizer_path = 'tfidf_vectorizerUMPOH.pkl'

# Cek file dataset utama
if not os.path.exists(dataset_asli):
    raise FileNotFoundError("❌ File dataset asli tidak ditemukan.")

# Baca dataset asli
df_asli = pd.read_csv(dataset_asli, encoding='ISO-8859-1', engine='python', on_bad_lines='warn')

if 'tweet' not in df_asli.columns or 'label' not in df_asli.columns:
    raise ValueError("Dataset asli harus memiliki kolom 'tweet' dan 'label'.")

# Siapkan data dari laporan pengguna
if os.path.exists(laporan_path):
    df_laporan = pd.read_csv(laporan_path)

    if all(col in df_laporan.columns for col in ['judul', 'isi', 'label_benar']):
        df_laporan['text'] = df_laporan['judul'] + ' ' + df_laporan['isi']
        df_laporan['label'] = df_laporan['label_benar'].map({'Valid': 0, 'Hoaks': 1})
        df_laporan = df_laporan[['text', 'label']]
    else:
        print("⚠️ Kolom dalam laporan tidak lengkap. Lewati laporan.")
        df_laporan = pd.DataFrame(columns=['text', 'label'])
else:
    print("ℹ️ Tidak ada laporan kesalahan ditemukan.")
    df_laporan = pd.DataFrame(columns=['text', 'label'])

# Gabungkan dataset
df_asli.rename(columns={'tweet': 'text'}, inplace=True)
df_all = pd.concat([df_asli[['text', 'label']], df_laporan], ignore_index=True)
df_all.dropna(inplace=True)

# Preprocessing
X = df_all['text'].astype(str)
y = df_all['label']

# Stopword removal
factory = StopWordRemoverFactory()
stopwords = factory.get_stop_words()

# Vectorization
vectorizer = TfidfVectorizer(stop_words=stopwords, max_features=5000)
X_vec = vectorizer.fit_transform(X)

# Model training
model = MultinomialNB()
model.fit(X_vec, y)

# Simpan model & vectorizer
with open(model_path, 'wb') as f:
    pickle.dump(model, f)

with open(vectorizer_path, 'wb') as f:
    pickle.dump(vectorizer, f)

print("✅ Model berhasil dilatih ulang dan disimpan.")
print(f"- Total data dilatih: {len(df_all)}")
if not df_laporan.empty:
    print(f"- Termasuk {len(df_laporan)} laporan dari pengguna.")
