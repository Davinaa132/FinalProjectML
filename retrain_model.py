# retrain_model.py

import pandas as pd
import re
import os
import pickle
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ----------------------
# Fungsi preprocessing
# ----------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s.,]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ----------------------
# Load data awal
# ----------------------
print("📥 Membaca dataset utama...")
try:
    df_awal = pd.read_csv("datasetUMPOHoax.csv")
    df_awal = df_awal[['judul', 'isi', 'label']]  # Pastikan kolom sesuai
    df_awal.dropna(inplace=True)
    df_awal['text'] = df_awal['judul'] + " " + df_awal['isi']
    df_awal['text'] = df_awal['text'].apply(clean_text)
    df_awal = df_awal[['text', 'label']]
    print(f"✅ Dataset utama: {len(df_awal)} baris")
except Exception as e:
    print(f"❌ Gagal membaca dataset utama: {e}")
    exit()

# ----------------------
# Load laporan kesalahan
# ----------------------
laporan_path = "laporan_kesalahan.csv"
df_laporan = pd.DataFrame()

if os.path.exists(laporan_path):
    print("📥 Membaca laporan kesalahan pengguna...")
    try:
        df_laporan = pd.read_csv(laporan_path)
        df_laporan = df_laporan[['judul', 'isi', 'label_benar']]
        df_laporan.dropna(inplace=True)
        df_laporan['text'] = (df_laporan['judul'] + " " + df_laporan['isi']).apply(clean_text)
        df_laporan['label'] = df_laporan['label_benar'].map({'Valid': 0, 'Hoaks': 1})
        df_laporan.dropna(subset=['label'], inplace=True)
        df_laporan['label'] = df_laporan['label'].astype(int)
        df_laporan = df_laporan[['text', 'label']]
        print(f"✅ Laporan ditambahkan: {len(df_laporan)} baris")
    except Exception as e:
        print(f"⚠️ Gagal membaca laporan kesalahan: {e}")
else:
    print("ℹ️ Tidak ada laporan kesalahan ditemukan.")

# ----------------------
# Gabungkan dan bersihkan data
# ----------------------
df_all = pd.concat([df_awal, df_laporan], ignore_index=True)
df_all.drop_duplicates(subset='text', inplace=True)
df_all.dropna(inplace=True)

if df_all['label'].nunique() < 2:
    print("❌ Gagal: Data tidak mencakup dua kelas berbeda (valid dan hoaks).")
    exit()

print(f"🧹 Total data setelah penggabungan dan pembersihan: {len(df_all)} baris")

# ----------------------
# TF-IDF & Pelatihan Model
# ----------------------
stopwords = StopWordRemoverFactory().get_stop_words()

vectorizer = TfidfVectorizer(stop_words=stopwords, max_features=5000)
