# -*- coding: utf-8 -*-
"""app.py

Script untuk deteksi berita hoaks berbasis Streamlit.
"""

import streamlit as st
import pickle
import os

# Ganti path ini jika file disimpan di folder lain
model_path = 'multinomial_nb_modelUMPOH.pkl'
vectorizer_path = 'tfidf_vectorizerUMPOH.pkl'

# Cek keberadaan file
if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
    st.error("❌ File model atau vectorizer tidak ditemukan di folder 'model/'. Jalankan train_model.py terlebih dahulu.")
    st.stop()

# Load model dan vectorizer
with open(model_path, 'rb') as f:
    model = pickle.load(f)

with open(vectorizer_path, 'rb') as f:
    vectorizer = pickle.load(f)

# Konfigurasi halaman
st.set_page_config(page_title="Deteksi Berita Hoaks", layout="centered")
st.title("📰 Deteksi Berita Hoaks Bahasa Indonesia")
st.markdown("Masukkan isi berita atau cuitan di bawah ini untuk mendeteksi apakah mengandung **hoaks** atau **tidak**.")

# Input pengguna
tweet = st.text_area("📝 Masukkan Teks Berita atau Tweet", height=200)

# Tombol deteksi
if st.button("🔍 Deteksi"):
    if tweet.strip() == "":
        st.warning("⚠️ Harap masukkan teks terlebih dahulu.")
    else:
        X_input = vectorizer.transform([tweet])
        prediction = model.predict(X_input)[0]
        proba = model.predict_proba(X_input)[0][prediction]

        if prediction == 1:
            st.error(f"🚨 Deteksi: **HOAKS** (Probabilitas: {proba:.2f})")
        else:
            st.success(f"✅ Deteksi: **VALID** (Probabilitas: {proba:.2f})")
