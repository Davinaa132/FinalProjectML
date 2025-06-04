# -*- coding: utf-8 -*-
"""app.py

Script untuk deteksi berita hoaks berbasis Streamlit.
"""

import streamlit as st
import pickle
import os

# Ganti path ini sesuai struktur folder saat diupload ke GitHub atau dijalankan lokal
model_path = 'multinomial_nb_model.pkl'
vectorizer_path = 'tfidf_vectorizer.pkl'

# Pastikan file path valid
if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
    st.error("❌ File model atau vectorizer tidak ditemukan. Pastikan file sudah ditempatkan di folder yang sesuai.")
    st.stop()

with open(model_path, 'rb') as f:
    model = pickle.load(f)

with open(vectorizer_path, 'rb') as f:
    vectorizer = pickle.load(f)

st.set_page_config(page_title="Deteksi Berita Hoaks", layout="centered")
st.title("📰 Deteksi Berita Hoaks Bahasa Indonesia")
st.markdown("Masukkan **judul** dan **isi berita**, lalu klik tombol **Deteksi** untuk mengetahui apakah berita itu **HOAKS (1)** atau **VALID (0)**.")

judul = st.text_input("📝 Judul Berita")
isi = st.text_area("📄 Isi Berita")

if st.button("🔍 Deteksi"):
    if judul and isi:
        full_text = judul + " " + isi
        X_input = vectorizer.transform([full_text])
        prediction = model.predict(X_input)[0]
        proba = model.predict_proba(X_input)[0][prediction]

        if prediction == 1:
            st.error(f"🚨 Deteksi: **HOAKS** (Probabilitas: {proba:.2f})")
        else:
            st.success(f"✅ Deteksi: **VALID** (Probabilitas: {proba:.2f})")
    else:
        st.warning("⚠️ Harap isi semua kolom terlebih dahulu.")
