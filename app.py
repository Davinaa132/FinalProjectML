# -*- coding: utf-8 -*-
"""app.py

Script Streamlit untuk deteksi berita hoaks berdasarkan judul dan URL.
"""

import streamlit as st
import pickle
import os
import requests
from bs4 import BeautifulSoup

# Daftar sumber berita resmi
sumber_resmi = [
    "cnnindonesia.com",
    "kompas.com",
    "tempo.co",
    "antaranews.com",
    "detik.com",
    "liputan6.com",
    "beritasatu.com",
    "bbc.com",
    "cnbcindonesia.com",
    "republika.co.id"
]

# Fungsi cek apakah URL berasal dari sumber resmi
def is_sumber_resmi(url):
    for domain in sumber_resmi:
        if domain in url:
            return True
    return False

# Ganti path model jika disimpan di folder lain
model_path = 'multinomial_nb_modelUMPOH.pkl'
vectorizer_path = 'tfidf_vectorizerUMPOH.pkl'

# Cek file model & vectorizer
if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
    st.error("❌ Model atau vectorizer tidak ditemukan. Pastikan sudah melatih model.")
    st.stop()

# Load model
with open(model_path, 'rb') as f:
    model = pickle.load(f)

with open(vectorizer_path, 'rb') as f:
    vectorizer = pickle.load(f)

# Fungsi untuk ambil isi artikel dari URL
def extract_article_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Ambil semua <p>
        paragraphs = soup.find_all('p')
        text = ' '.join([p.get_text() for p in paragraphs])

        return text.strip()
    except Exception as e:
        return f"[Gagal mengambil isi dari URL: {e}]"

# Konfigurasi Streamlit
st.set_page_config(page_title="Deteksi Hoaks Berita", layout="centered")
st.title("📰 Deteksi Hoaks dari Judul dan URL Berita")
st.markdown("Masukkan **judul** dan **tautan URL** berita. Sistem akan mendeteksi apakah berita tersebut hoaks atau valid.")

# Input
judul = st.text_input("📝 Judul Berita")
url = st.text_input("🔗 URL Berita")

# Tombol Deteksi
if st.button("🔍 Deteksi"):
    if not judul or not url:
        st.warning("⚠️ Silakan isi judul dan URL berita terlebih dahulu.")
    else:
        st.info("📡 Mengambil isi berita dari URL...")
        isi = extract_article_from_url(url)

        if isi.startswith("[Gagal"):
            st.error(isi)
        else:
            full_text = judul + " " + isi
            X_input = vectorizer.transform([full_text])
            prediction = model.predict(X_input)[0]

            # Tangani kemungkinan error predict_proba
            try:
                proba_array = model.predict_proba(X_input)[0]
                prob_valid = proba_array[0]
                prob_hoax = proba_array[1]
            except Exception:
                prob_valid = 0.0
                prob_hoax = 0.0

            # Threshold klasifikasi valid minimal 40%
            threshold_valid = 0.40

            if prob_valid >= threshold_valid:
                st.success(f"✅ Deteksi: **VALID** (Probabilitas: {prob_valid:.2f})")
            else:
                st.error(f"🚨 Deteksi: **HOAKS** (Probabilitas: {prob_hoax:.2f})")

                # Peringatan jika dari sumber resmi
                if is_sumber_resmi(url):
                    st.info("⚠️ *Hasil ini mungkin tidak akurat karena berita berasal dari sumber resmi. Pertimbangkan untuk memverifikasi secara manual.*")

            # Tampilkan probabilitas untuk transparansi
            st.markdown("### 📊 Probabilitas Klasifikasi:")
            st.markdown(f"- **Valid:** {prob_valid:.2f}")
            st.markdown(f"- **Hoaks:** {prob_hoax:.2f}")
